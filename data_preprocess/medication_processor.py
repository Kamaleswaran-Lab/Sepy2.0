from pathlib import Path
import pandas as pd
import numpy as np
import sys
sys.path.append("../")
import tqdm

import re
from typing import Optional, Dict
import med_processing_utils as mpu
import parameter_extractor as pe

### NOT RECORDED are the ones with med stop!!!!
## Should impute total volume in the first hour if meds are recorded before supertable time index 

def calculate_actual_infusion_duration(start_row, all_med_rows, processed_indices):
    """
    Calculate actual infusion duration based on "Infuse" actions.
    
    Args:
        start_row: Row with "Begin Bag" or "Rate Change" action that starts the infusion
        all_med_rows: All rows for this order_med_id + formulary_name, sorted by time
        processed_indices: Set of row indices that have already been processed
        
    Returns:
        tuple: (duration_hours, list_of_used_infuse_indices, last_infuse_time)
    """
    all_med_rows['med_action_time'] = pd.to_datetime(all_med_rows['med_action_time'])
    start_time = start_row["med_action_time"]
    start_action = start_row["med_action"]
    
    # Find subsequent rows for this medication that haven't been processed
    subsequent_rows = all_med_rows[
        (all_med_rows.med_action_time > start_time)
    ].sort_values('med_action_time')
    
    last_infuse_time = None
    used_infuse_indices = []
    
    for _, next_row in subsequent_rows.iterrows():
        if next_row["med_action"] == "Infuse":
            last_infuse_time = next_row["med_action_time"]
            used_infuse_indices.append(next_row.name)
            print(f"counting infusion at time {last_infuse_time}")
        elif (next_row["med_action"] == "Begin Bag") or (next_row["med_action"] == "Waste"):
            # Stop when we hit the next bag - this ends the current infusion period
            print(f"stopping infusion at time {last_infuse_time}")
            break
        elif next_row["med_action"] == "Rate Change":
            if start_action == "Rate Change":
                # Rate Change followed by another Rate Change - leave for further processing
                print(f"Rate Change followed by another Rate Change - leaving for further processing")
                return 0.0, [], None
            else:
                # Begin Bag followed by Rate Change - this ends the current infusion period
                print(f"Begin Bag followed by Rate Change - ending infusion period")
                break
    
    if last_infuse_time:
        # Calculate actual duration based on last infuse time
        print(f"last infuse time: {last_infuse_time}, start time: {start_time}")
        raw_duration = pd.to_datetime(last_infuse_time) - pd.to_datetime(start_time)
        duration_hours = raw_duration.total_seconds() / 3600.0
        return duration_hours, used_infuse_indices, last_infuse_time
    else:
        # No infuse actions found - infusion may have been stopped immediately
        action_desc = "Rate Change" if start_action == "Rate Change" else "Begin Bag"
        print(f"WARNING: No 'Infuse' actions found after '{action_desc}' at {start_time}")
        return 0.0, [], None

def _calculate_actual_duration_and_update(row, med_rows, processed_indices):
    """
    Calculate actual duration from Infuse actions and update processed indices.
    
    Returns:
        tuple: (actual_duration, last_infuse_time, updated_processed_indices)
    """
    actual_duration, used_infuse_indices, last_infuse_time = calculate_actual_infusion_duration(
        row, med_rows, processed_indices
    )
    processed_indices.update(used_infuse_indices)
    return actual_duration, last_infuse_time, processed_indices


def _initialize_medication_params(medsdict, order_id, med_name, volume, rate, duration, med_start, med_stop):
    """
    Initialize medication parameters in medsdict (first time setup).
    """
    medsdict[order_id][med_name]["volume"] = [volume]
    medsdict[order_id][med_name]["rate"] = [rate]
    medsdict[order_id][med_name]["duration"] = [duration]
    medsdict[order_id][med_name]["med_start"] = [med_start]
    medsdict[order_id][med_name]["med_stop"] = [med_stop]
    medsdict[order_id][med_name]["set"] = True
    
    # Store original parameters for fallback
    medsdict[order_id][med_name]["original_rate"] = rate
    medsdict[order_id][med_name]["original_volume"] = volume
    medsdict[order_id][med_name]["original_duration"] = duration


def _append_medication_params(medsdict, order_id, med_name, volume, rate, duration, med_start, med_stop):
    """
    Append medication parameters to existing medsdict entry.
    """
    medsdict[order_id][med_name]["volume"].append(volume)
    medsdict[order_id][med_name]["rate"].append(rate)
    medsdict[order_id][med_name]["duration"].append(duration)
    medsdict[order_id][med_name]["med_start"].append(med_start)
    medsdict[order_id][med_name]["med_stop"].append(med_stop)


def _recalculate_rate_if_1hr_assumption(params, volume, rate, final_duration, row, supertable):
    """
    Recalculate rate if volume exists and rate was based on 1hr assumption for small fluids.
    
    Returns:
        float: Recalculated rate or original rate
        
    """
    #TODO: Check if removing is_fluid check is correct
    if pd.notna(volume) and final_duration > 0:
        # Get infusion params to check if it's a fluid
        nearest_idx = supertable.index.get_indexer([row["med_action_time"]], method='nearest')[0]
        weight = supertable['daily_weight_kg'].values[nearest_idx]
        #infusion_params = pe.extract_infusion_params(row, weight)
        
        # If we have volume and either no rate or the rate was calculated from 1-hour assumption
        if (pd.isna(rate) or 
            (pd.notna(params.get("duration")) and params["duration"] == 1.0 and volume <= 1000)):
            
            new_rate = volume / final_duration
            print(f"Recalculated rate using actual duration: {new_rate:.1f}mL/h (was {rate}mL/h from 1-hour assumption)")
            return new_rate
    
    return rate


def _determine_duration_and_stop_time(actual_duration, last_infuse_time, params, med_start):
    """
    Determine final duration and med_stop time, preferring actual over theoretical.
    
    Returns:
        tuple: (final_duration, med_stop)
    """
    if actual_duration > 0:
        final_duration = actual_duration
        med_stop = last_infuse_time
        print(f"Using actual duration: {final_duration:.2f} hours")
    else:
        final_duration = params.get("duration", 1.0) if pd.notna(params.get("duration")) else 1.0
        med_stop = med_start + pd.Timedelta(hours=final_duration)
        print(f"Using theoretical duration: {final_duration} hours")
    
    return final_duration, med_stop


def handle_begin_bag(row, med_rows, processed_indices, order_id, med_name, medsdict, supertable, 
                     is_explicit=False, action_label="BEGIN BAG", params=None):
    """
    Unified function to handle all Begin Bag scenarios (explicit and implicit).
    
    Args:
        row: Current medication action row
        med_rows: All rows for this medication
        processed_indices: Set of already processed row indices
        order_id: Medication order ID
        med_name: Medication name
        medsdict: Dictionary storing medication parameters
        supertable: Patient data table for weight lookups
        is_explicit: True for explicit "Begin Bag" actions, False for implicit (Rate Change, Infuse, etc.)
        action_label: Label for logging (e.g., "BEGIN BAG", "IMPLICIT BEGIN BAG", "RATE CHANGE")
        params: Optional pre-extracted parameters (for dilution pairs); if None, will extract from row
    
    Returns:
        tuple: (updated_medsdict, updated_processed_indices)
    """
    is_initial = not medsdict[order_id][med_name]["set"]
    
    if is_initial:
        print(f"Processing initial bag for {action_label}")
    else:
        print(f"Processing subsequent bag for {action_label}")
    
    # Get theoretical parameters for this action
    if params is None:
        params = pe.get_order_params_for_row(row, supertable)
        
    # Calculate actual duration from Infuse actions
    actual_duration, last_infuse_time, processed_indices = _calculate_actual_duration_and_update(
        row, med_rows, processed_indices
    )
    
    # Determine final duration and med_stop (use actual duration if not zero otherwise duration from params if not None otherwise 1hr)
    final_duration, med_stop = _determine_duration_and_stop_time(
        actual_duration, last_infuse_time, params, params["med_start"]
    )
    
    # Handle special case: medication stop (rate=0) [happens only when the med_action_dose_unit is not NAN and the rate is 0 (when rate is changed to zero)]
    if pd.notna(params.get("rate")) and params["rate"] == 0:
        volume = 0
        rate = 0
        final_duration = 0
        med_stop = params["med_start"]
        print(f"Medication stopped at time {params['med_start']}")
    
    # Try to get valid parameters
    elif params["final_check"]:
        # Use extracted parameters
        volume = params["volume"]
        #If volume could not be extracted from params, but rate and duration have been extracted, then calculate volume
        if pd.isna(volume) and pd.notna(params["rate"]) and pd.notna(params["duration"]):
            volume = params["rate"] * params["duration"]
        #Now volume , rate and duration exist
        assert pd.notna(volume), "If final check was true, volume should be nan at this point"
        
        #Now, recalculate the rate using final_duration (actual infused duration)
        rate = _recalculate_rate_if_1hr_assumption(params, volume, params["rate"], final_duration, row, supertable)
        print(f"Using extracted parameters: Volume={volume:.1f}mL, Rate={rate:.1f}mL/h")
    
    elif not is_initial:
        #If final check was not True, then use original order's parameters if this is a repeat recording of the 
        # same order number and med name
        # Subsequent bag (explicit or implicit) with failed extraction - use original stored params
        volume = medsdict[order_id][med_name]["original_volume"]
        rate = medsdict[order_id][med_name]["original_rate"]
        print(f"Using original parameters as fallback: Volume={volume:.1f}mL, Rate={rate:.1f}mL/h")
    
    else:
        # Initial bag with failed extraction - try defaults
        print("Parameter extraction failed - checking defaults")
        volume = None
        rate = None
        
        # Try volume_from_concentration first
        if 'volume_from_concentration' in row and pd.notna(row['volume_from_concentration']):
            volume = row['volume_from_concentration']
            rate = volume / final_duration if final_duration > 0 else None
            print(f"Using volume_from_concentration: {volume:.1f}mL, calculated rate: {rate:.1f}mL/h")
        
        # Try rate_default_numeric if volume not found
        elif 'rate_default_numeric' in row and pd.notna(row['rate_default_numeric']):
            rate = row['rate_default_numeric']
            volume = rate * final_duration
            print(f"Using rate_default_numeric: {rate:.1f}mL/h, calculated volume: {volume:.1f}mL")
        
        # Check if we got valid parameters
        if volume is None or rate is None or pd.isna(volume) or pd.isna(rate):
            print(f"No valid parameters available - skipping {action_label}")
            return medsdict, processed_indices
    
    # Store parameters (initialize or append)
    if is_initial:
        _initialize_medication_params(medsdict, order_id, med_name, volume, rate, final_duration, 
                                     params["med_start"], med_stop)
        print(f"{action_label} (INITIAL) - Final params: Start={params['med_start']}, Stop={med_stop}, "
              f"Rate={rate:.1f}mL/h, Volume={volume:.1f}mL, Duration={final_duration:.2f}h")
    else:
        _append_medication_params(medsdict, order_id, med_name, volume, rate, final_duration, 
                                 params["med_start"], med_stop)
        print(f"{action_label} (SUBSEQUENT) - Final params: Start={params['med_start']}, Stop={med_stop}, "
              f"Rate={rate:.1f}mL/h, Volume={volume:.1f}mL, Duration={final_duration:.2f}h")
    
    return medsdict, processed_indices


def handle_implicit_begin_bag(row, med_rows, processed_indices, order_id, med_name, medsdict, supertable, params=None):
    """
    Handle case where first action is not Begin Bag or orphaned rate change - create implicit infusion period.
    This is a wrapper that calls the unified handle_begin_bag function.
    
    Args:
        row: Current row (Infuse or Rate Change)
        med_rows: All rows for this medication
        processed_indices: Set of already processed indices
        order_id: Order ID
        med_name: Medication name
        medsdict: Medication dictionary
        supertable: Patient data table
        params: Optional pre-extracted parameters (for dilution pairs); if None, will extract from row
    
    Returns:
        tuple: (updated_medsdict, updated_processed_indices)
    """
    return handle_begin_bag(row, med_rows, processed_indices, order_id, med_name, medsdict, supertable,
                           is_explicit=False, action_label="IMPLICIT BEGIN BAG", params=params)


def handle_rate_change(row, med_rows, processed_indices, order_id, med_name, medsdict, supertable, params=None):
    # Get theoretical parameters for this rate change
    if params is None:
        params = pe.get_order_params_for_row(row, supertable)
        
    # Get previous infusion details
    prev_start = medsdict[order_id][med_name]["med_start"][-1]
    prev_stop = medsdict[order_id][med_name]["med_stop"][-1]
    prev_rate = medsdict[order_id][med_name]["rate"][-1]
    prev_volume = medsdict[order_id][med_name]["volume"][-1]
    prev_duration = medsdict[order_id][med_name]["duration"][-1]
    
    # Get original parameters for fallback
    original_rate = medsdict[order_id][med_name]["original_rate"]
    original_volume = medsdict[order_id][med_name]["original_volume"]
    original_duration = medsdict[order_id][med_name]["original_duration"]
    
    # Determine new parameters (use original as fallback)
    if params["final_check"]:
        new_rate = params["rate"]
        new_volume = params["volume"]
        new_duration = params["duration"]
    elif (params["rate"] == 0) & (params["volume"] == 0) & (params["duration"] == 0):
        new_rate = 0
        new_volume = 0
        new_duration = 0
    else:
        new_rate = original_rate
        new_volume = original_volume  
        new_duration = original_duration
        print("Rate change params failed validation - using original parameters")
    
    # Check if rate actually changed
    if pd.notna(new_rate) and pd.notna(prev_rate) and new_rate == prev_rate:
        print("Rate change but no actual rate difference - ignoring")
        return medsdict, processed_indices
        
    # Check if rate change is valid (allow 0 for medication stop, reject negative)
    if pd.isna(new_rate) or new_rate < 0:
        print("Invalid new rate - ignoring rate change")
        return medsdict, processed_indices
        
        
    # Check timing relative to previous infusion
    med_action_time = row["med_action_time"]
    print(f"The current med action time is : {med_action_time}")
    if med_action_time < prev_stop:
        # Rate change during active infusion - split the period
        print(f"Rate change during active infusion (was due to end at {prev_stop})")
        
        # Calculate what was already delivered
        time_elapsed = (med_action_time - prev_start).total_seconds() / 3600.0
        volume_delivered = prev_rate * time_elapsed
        remaining_volume = prev_volume - volume_delivered
        
        print(f"Time elapsed: {time_elapsed:.2f}h, Volume delivered: {volume_delivered:.1f}mL, Remaining: {remaining_volume:.1f}mL")
        
        # Update previous period to end at rate change time
        medsdict[order_id][med_name]["med_stop"][-1] = med_action_time
        medsdict[order_id][med_name]["duration"][-1] = time_elapsed
        
        # Calculate actual duration for new period using Infuse actions
        actual_duration, used_infuse_indices, last_infuse_time = calculate_actual_infusion_duration(
            row, med_rows, processed_indices
        )
        processed_indices.update(used_infuse_indices)
        
        # Determine final duration for new period
        if new_rate == 0:
            # Medication stopped - use actual duration from infuses, volume is remaining volume
            if actual_duration > 0:
                final_duration = actual_duration
                new_stop_time = last_infuse_time
                print(f"Medication stopped - using actual duration from Infuse actions: {final_duration:.2f}h")
            else:
                # No infuse actions found - medication stopped immediately
                final_duration = 0
                new_stop_time = med_action_time
                print(f"Medication stopped immediately at time {med_action_time}")
        elif actual_duration > 0:
            final_duration = actual_duration
            new_stop_time = last_infuse_time
            print(f"Using actual duration from Infuse actions: {final_duration:.2f}h")
        else:
            # No infuse actions found - calculate theoretical duration with remaining volume
            if remaining_volume > 0 and new_rate > 0:
                final_duration = remaining_volume / new_rate
                new_stop_time = med_action_time + pd.Timedelta(hours=final_duration)
                print(f"No Infuse actions found - using theoretical duration: {final_duration:.2f}h")
            else:
                print("Cannot calculate duration - skipping rate change")
                return medsdict, processed_indices
        
        # Add new period with remaining volume and new rate
        _append_medication_params(medsdict, order_id, med_name, remaining_volume, new_rate, final_duration, med_action_time, new_stop_time)
        print(f"RATE CHANGE (DURING INFUSION) - Final params: Start={med_action_time}, Stop={new_stop_time}, Rate={new_rate:.1f}mL/h, Volume={remaining_volume:.1f}mL, Duration={final_duration:.2f}h")
    else:
        # Rate change after previous infusion ended - treat as orphaned rate change (implicit begin bag)
        print(f"Orphaned rate change after infusion ended (ended at {prev_stop}) - treating as implicit Begin Bag")
        
        # Use the enhanced handle_implicit_begin_bag function
        medsdict, processed_indices = handle_implicit_begin_bag(
            row, med_rows, processed_indices, order_id, med_name, medsdict, supertable
        )

    return medsdict, processed_indices

def handle_not_recorded(row, processed_indices, order_id, med_name, medsdict, supertable, params=None):
    # Check if med_stop is available (it should be for "Not Recorded" actions)
    if pd.isna(row["med_stop"]):
        print("Not Recorded action missing med_stop - this is unexpected, ignoring")
        return medsdict, processed_indices
    
    med_start_time = row["med_action_time"]
    med_stop_time = pd.to_datetime(row["med_stop"])
    duration = (med_stop_time - med_start_time).total_seconds() / 3600.0
    
    print(f"Not Recorded with med_start: {med_start_time}, med_stop: {med_stop_time}, duration: {duration:.2f}h")
    
    # Get order parameters for this row
    if params is None:
        params = pe.get_order_params_for_row(row, supertable)
    
    if params["final_check"]:
        print("Valid params found for Not Recorded action - using calculated parameters")
        
        # Use calculated parameters with actual med_stop time
        volume = params["volume"]
        rate = params["rate"]
        
        # Calculate volume if missing using actual duration
        if pd.isna(volume) and pd.notna(rate):
            volume = rate * duration
            print(f"Calculated volume from rate and actual duration: {volume:.1f}mL")
        
        if not medsdict[order_id][med_name]["set"]:
            _initialize_medication_params(medsdict, order_id, med_name, volume, rate, duration, med_start_time, med_stop_time)
            print(f"NOT RECORDED (INITIAL) - Final params: Start={med_start_time}, Stop={med_stop_time}, Rate={rate:.1f}mL/h, Volume={volume:.1f}mL, Duration={duration:.2f}h")
        else:
            _append_medication_params(medsdict, order_id, med_name, volume, rate, duration, med_start_time, med_stop_time)
            print(f"NOT RECORDED (SUBSEQUENT) - Final params: Start={med_start_time}, Stop={med_stop_time}, Rate={rate:.1f}mL/h, Volume={volume:.1f}mL, Duration={duration:.2f}h")
            
    else:
        print("Invalid params for Not Recorded action - checking if medication already set")
        
        if medsdict[order_id][med_name]["set"]:
            # Use existing parameters with this row's times
            original_rate = medsdict[order_id][med_name]["original_rate"]
            original_volume = medsdict[order_id][med_name]["original_volume"]
            
            # Calculate volume using actual duration and original rate
            if pd.notna(original_rate):
                volume = original_rate * duration
                print(f"Using original rate {original_rate:.1f}mL/h with actual duration {duration:.2f}h -> volume {volume:.1f}mL")
            else:
                volume = original_volume if pd.notna(original_volume) else None
                print(f"Using original volume: {volume}")
            
            # Add new period with original parameters and actual times
            _append_medication_params(medsdict, order_id, med_name, volume, original_rate, duration, med_start_time, med_stop_time)
            print(f"NOT RECORDED (FALLBACK) - Final params: Start={med_start_time}, Stop={med_stop_time}, Rate={original_rate:.1f}mL/h, Volume={volume:.1f}mL, Duration={duration:.2f}h")
        else:
            # No existing parameters - try using concentration_default or rate_default as last resort
            print("No existing parameters - checking for concentration_default or rate_default")
            
            volume = None
            rate = None
            
            # Try volume_from_concentration first
            if 'volume_from_concentration' in row and pd.notna(row['volume_from_concentration']):
                volume = row['volume_from_concentration']
                rate = volume / duration if duration > 0 else None
                print(f"Using volume_from_concentration: {volume:.1f}mL, calculated rate: {rate:.1f}mL/h")
            
            # Try rate_default_numeric if volume not found
            elif 'rate_default_numeric' in row and pd.notna(row['rate_default_numeric']):
                rate = row['rate_default_numeric']
                volume = rate * duration
                print(f"Using rate_default_numeric: {rate:.1f}mL/h, calculated volume: {volume:.1f}mL")
            
            # If we got either volume or rate, proceed
            if volume is not None and rate is not None and pd.notna(volume) and pd.notna(rate):
                _initialize_medication_params(medsdict, order_id, med_name, volume, rate, duration, med_start_time, med_stop_time)
                print(f"NOT RECORDED (DEFAULT) - Final params: Start={med_start_time}, Stop={med_stop_time}, Rate={rate:.1f}mL/h, Volume={volume:.1f}mL, Duration={duration:.2f}h")
            else:
                print("No existing parameters, invalid params, and no defaults available - ignoring")

    return medsdict, processed_indices


def handle_bolus(row, processed_indices, order_id, med_name, medsdict, supertable, params=None):
    # Get order parameters for this bolus
    if params is None:
        params = pe.get_order_params_for_row(row, supertable)
    med_start_time = row["med_action_time"]
    
    if params["final_check"]:
        print("Valid params found for Bolus - using calculated parameters")
        
        # Use calculated parameters
        volume = params["volume"]
        rate = params["rate"]
        duration = params["duration"]
        
        # Calculate volume if missing
        if pd.isna(volume) and pd.notna(rate) and pd.notna(duration):
            volume = rate * duration
            
        med_stop_time = med_start_time + pd.Timedelta(hours=duration)
        
    else:
        print("Invalid params for Bolus - using fallback logic")
        
        # Fallback logic: assume 1 hour duration, use volume if available
        duration = 1.0  # 1 hour default for bolus
        
        # Check if volume is available from params (even though final_check failed)
        if pd.notna(params["volume"]):
            volume = params["volume"]
            rate = volume / duration  # Calculate rate from volume and 1-hour duration
            print(f"Using available volume {volume:.1f}mL with 1-hour duration -> rate {rate:.1f}mL/h")
        else:
            # Check if medication already set up to use original volume
            if medsdict[order_id][med_name]["set"]:
                original_volume = medsdict[order_id][med_name]["original_volume"]
                original_rate = medsdict[order_id][med_name]["original_rate"]
                
                if pd.notna(original_volume):
                    volume = original_volume
                    rate = volume / duration
                    print(f"Using original volume {volume:.1f}mL with 1-hour duration -> rate {rate:.1f}mL/h")
                elif pd.notna(original_rate):
                    rate = original_rate
                    volume = rate * duration
                    print(f"Using original rate {rate:.1f}mL/h with 1-hour duration -> volume {volume:.1f}mL")
                else:
                    print("No original parameters available for Bolus fallback - checking defaults")
                    # Try defaults as last resort
                    if 'volume_from_concentration' in row and pd.notna(row['volume_from_concentration']):
                        volume = row['volume_from_concentration']
                        rate = volume / duration
                        print(f"Using volume_from_concentration: {volume:.1f}mL -> rate {rate:.1f}mL/h")
                    elif 'rate_default_numeric' in row and pd.notna(row['rate_default_numeric']):
                        rate = row['rate_default_numeric']
                        volume = rate * duration
                        print(f"Using rate_default_numeric: {rate:.1f}mL/h -> volume {volume:.1f}mL")
                    else:
                        print("No defaults available - ignoring")
                        return medsdict, processed_indices
            else:
                # No existing parameters - try defaults
                print("No existing parameters for Bolus - checking defaults")
                if 'volume_from_concentration' in row and pd.notna(row['volume_from_concentration']):
                    volume = row['volume_from_concentration']
                    rate = volume / duration
                    print(f"Using volume_from_concentration: {volume:.1f}mL -> rate {rate:.1f}mL/h")
                elif 'rate_default_numeric' in row and pd.notna(row['rate_default_numeric']):
                    rate = row['rate_default_numeric']
                    volume = rate * duration
                    print(f"Using rate_default_numeric: {rate:.1f}mL/h -> volume {volume:.1f}mL")
                else:
                    print("No defaults available - ignoring Bolus")
                    return medsdict, processed_indices
                
        med_stop_time = med_start_time + pd.Timedelta(hours=duration)
    
    # Add bolus to medsdict
    if not medsdict[order_id][med_name]["set"]:
        _initialize_medication_params(medsdict, order_id, med_name, volume, rate, duration, med_start_time, med_stop_time)
        print(f"BOLUS (INITIAL) - Final params: Start={med_start_time}, Stop={med_stop_time}, Rate={rate:.1f}mL/h, Volume={volume:.1f}mL, Duration={duration:.2f}h")
    else:
        _append_medication_params(medsdict, order_id, med_name, volume, rate, duration, med_start_time, med_stop_time)
        print(f"BOLUS (SUBSEQUENT) - Final params: Start={med_start_time}, Stop={med_stop_time}, Rate={rate:.1f}mL/h, Volume={volume:.1f}mL, Duration={duration:.2f}h")
    
    return medsdict, processed_indices

def process_medication_timeline_new(order_id, med_name, all_order_rows, supertable, medsdict):
    """
    NEW FUNCTION: Process medication timeline chronologically with infusion duration validation.
    
    Args:
        order_id: Order ID being processed
        med_name: Medication name being processed  
        all_order_rows: All rows for this order_id, sorted by time
        supertable: Patient data table
        medsdict: Medication dictionary to update
        
        tuple: (updated_medsdict, processed_row_indices)
    """
    # Get rows for this specific medication
    med_rows = all_order_rows[all_order_rows['med_name'] == med_name].sort_values('med_action_time')
    processed_indices = set()
    orphaned_infuse_rows = []
    
    print(f"Processing {med_name} with {len(med_rows)} rows")
    med_rows['med_action_time'] = pd.to_datetime(med_rows['med_action_time'])
    for _, row in med_rows.iterrows():
        if row.name in processed_indices:
            continue
            
        med_action = row["med_action"]
        med_action_time = row["med_action_time"]
        
        print(f"Processing {med_action} at {med_action_time}")
        
        if med_action == "Begin Bag":
            processed_indices.add(row.name)
            
            # Use unified Begin Bag handler
            medsdict, processed_indices = handle_begin_bag(
                row, med_rows, processed_indices, order_id, med_name, medsdict, supertable,
                is_explicit=True, action_label="BEGIN BAG"
            )
                    
        elif med_action == "Rate Change":
            processed_indices.add(row.name)
            print("Processing Rate Change")
            
            # Get current medication state
            if not medsdict[order_id][med_name]["set"]:
                print("First action is Rate Change - treating as implicit Begin Bag")
                medsdict, processed_indices = handle_implicit_begin_bag(
                    row, med_rows, processed_indices, order_id, med_name, medsdict, supertable
                )
                continue
            
            medsdict, processed_indices = handle_rate_change(
                row, med_rows, processed_indices, order_id, med_name, medsdict, supertable
            )
                
        elif med_action == "Infuse":
            # Check if this is the first action and no medication is set up yet
            if not medsdict[order_id][med_name]["set"] and row.name not in processed_indices:
                # Check if there's a "Begin Bag" within one hour after this Infuse
                upcoming_begin_bag = None
                for _, future_row in med_rows.iterrows():
                    if future_row["med_action_time"] < row["med_action_time"]: #TODO: check others - removed equal to 
                        continue
                    future_time = future_row["med_action_time"]
                    time_diff = (future_time - med_action_time).total_seconds() / 3600.0
                    
                    if future_row["med_action"] == "Begin Bag" and time_diff <= 1.0:
                        upcoming_begin_bag = future_row
                        break
                    elif time_diff > 1.0:
                        break  # No point checking further
                
                if upcoming_begin_bag is not None:
                    print(f"Infuse at {med_action_time} has Begin Bag within 1 hour at {upcoming_begin_bag['med_action_time']} - ignoring this Infuse")
                    processed_indices.add(row.name)
                    continue
                else:
                    print("First action is Infuse with no upcoming Begin Bag - treating as implicit Begin Bag")
                    medsdict, processed_indices = handle_implicit_begin_bag(
                        row, med_rows, processed_indices, order_id, med_name, medsdict, supertable
                    )
                    continue
                
            # Check if this is an orphaned infuse action
            if row.name not in processed_indices:
                # This infuse wasn't consumed by any Begin Bag
                current_time = med_action_time
                
                # Check if there's an active infusion that should cover this time
                active_infusion_found = False
                if medsdict[order_id][med_name]["set"]:
                    for i, (start, stop) in enumerate(zip(
                        medsdict[order_id][med_name]["med_start"],
                        medsdict[order_id][med_name]["med_stop"]
                    )):
                        if pd.notna(start) and pd.notna(stop):
                            if start <= current_time <= stop:
                                active_infusion_found = True
                                break
                
                if not active_infusion_found:
                    # This is an orphaned infuse - treat as implicit begin bag
                    print(f"ORPHANED INFUSE: {med_name} at {current_time} - treating as implicit Begin Bag")
                    medsdict, processed_indices = handle_implicit_begin_bag(
                        row, med_rows, processed_indices, order_id, med_name, medsdict, supertable
                    )
                    continue
                else:
                    # This infuse is within an active period, mark as processed
                    processed_indices.add(row.name)
        
        elif med_action == "Not Recorded":
            processed_indices.add(row.name)
            print("Processing Not Recorded action")
            
            medsdict, processed_indices = handle_not_recorded(
                row, processed_indices, order_id, med_name, medsdict, supertable
            )

        elif med_action == "Bolus":
            processed_indices.add(row.name)
            print("Processing Bolus action")
            
            medsdict, processed_indices = handle_bolus(
                row, processed_indices, order_id, med_name, medsdict, supertable
            )
        else:
            # Handle other unrecorded med actions (not "Begin Bag", "Rate Change", "Infuse", "Not Recorded", or "Bolus")
            if not medsdict[order_id][med_name]["set"] and row.name not in processed_indices:
                print(f"Other unrecorded med action: {med_action} - checking if params are valid")
                params = pe.get_order_params_for_row(row, supertable)
                
                if params["final_check"]:
                    print(f"Valid params found for unrecorded action {med_action} - treating as implicit Begin Bag")
                    medsdict, processed_indices = handle_implicit_begin_bag(
                        row, med_rows, processed_indices, order_id, med_name, medsdict, supertable
                    )
                    continue
                else:
                    print(f"Invalid params for unrecorded action {med_action} - ignoring")
                    processed_indices.add(row.name)
            else:
                print(f"Unrecorded med action {med_action} after medication already set - ignoring")
                processed_indices.add(row.name)
    
    if orphaned_infuse_rows:
        print(f"Found {len(orphaned_infuse_rows)} orphaned infuse rows for {med_name}")
        for orphan in orphaned_infuse_rows:
            print(f"  - Orphaned at {orphan['time']}")
    
    # Summary logging for this medication
    if medsdict[order_id][med_name]["set"]:
        num_periods = len(medsdict[order_id][med_name]["med_start"])
        print(f"\n=== MEDICATION SUMMARY: {med_name} ===")
        print(f"Total infusion periods: {num_periods}")
        for i in range(num_periods):
            start = medsdict[order_id][med_name]["med_start"][i]
            stop = medsdict[order_id][med_name]["med_stop"][i]
            rate = medsdict[order_id][med_name]["rate"][i]
            volume = medsdict[order_id][med_name]["volume"][i]
            duration = medsdict[order_id][med_name]["duration"][i]
            print(f"  Period {i+1}: {start} to {stop} | Rate: {rate:.1f}mL/h | Volume: {volume:.1f}mL | Duration: {duration:.2f}h")
        print("=====================================\n")
    else:
        print(f"\n=== MEDICATION SUMMARY: {med_name} ===")
        print("No valid infusion periods established")
        print("=====================================\n")
    
    return medsdict, processed_indices
















def process_dilution_pairs(
    order_id, 
    fluid_med_name, 
    non_fluid_med_name, 
    all_order_rows, 
    supertable
):
    """
    Process medication-fluid dilution pairs where one medication is diluted in a fluid.
    
    Args:
        order_id: Order ID being processed
        fluid_med_name: Name of the fluid medication (is_fluid=True)
        non_fluid_med_name: Name of the non-fluid medication (is_fluid=False) 
        all_order_rows: All rows for this order_id, sorted by time
        supertable: Patient data table
        
    Returns:
        dict: Single medsdict entry for the combined medication
    """
    print(f"Processing dilution pair: {non_fluid_med_name} + {fluid_med_name}")
    
    # Get rows for both medications
    fluid_rows = all_order_rows[all_order_rows['med_name'] == fluid_med_name].sort_values('med_action_time')
    non_fluid_rows = all_order_rows[all_order_rows['med_name'] == non_fluid_med_name].sort_values('med_action_time')
    
    # Initialize combined medication dictionary
    combined_name = f"{non_fluid_med_name} + {fluid_med_name}"
    medsdict = {
        order_id: {
            combined_name: {
                'rate': [], 'duration': [], 'med_start': [], 'med_stop': [], 'volume': [], 'set': False,
                'original_rate': None, 'original_volume': None, 'original_duration': None
            }
        }
    }
    
    processed_indices = set()
    
    # Get all unique timestamps where both medications have actions
    fluid_times = set(fluid_rows['med_action_time'])
    non_fluid_times = set(non_fluid_rows['med_action_time'])
    common_times = fluid_times.intersection(non_fluid_times)
    non_common_times = (fluid_times.difference(common_times)).union(non_fluid_times.difference(common_times))

    if not common_times:
        print(f"ERROR: No synchronized timestamps found between {fluid_med_name} and {non_fluid_med_name}")
        return None
    
    if not non_common_times:
        print(f"FYI - there are {len(non_common_times)} non common times.")
    
    print(f"Found {len(common_times)} synchronized timestamps")
    
    # Process each synchronized timestamp
    for action_time in sorted(common_times):
        print(f"\nProcessing synchronized time: {action_time}")
        
        # Get rows for this timestamp
        fluid_time_rows = fluid_rows[fluid_rows['med_action_time'] == action_time]
        non_fluid_time_rows = non_fluid_rows[non_fluid_rows['med_action_time'] == action_time]
        
        # Process each pair of actions at this timestamp
        for _, fluid_row in fluid_time_rows.iterrows():
            for _, non_fluid_row in non_fluid_time_rows.iterrows():
                
                if fluid_row.name in processed_indices or non_fluid_row.name in processed_indices:
                    continue
                    
                print(f"Processing pair: {fluid_row['med_action']} (fluid) + {non_fluid_row['med_action']} (medication)")
                
                # Get parameters for both medications
                fluid_params = pe.get_order_params_for_row(fluid_row, supertable)
                non_fluid_params = pe.get_order_params_for_row(non_fluid_row, supertable)
                
                # Determine which parameters to use based on final_check priority
                primary_params = None
                secondary_params = None
                rate_source = None
                volume = None
                
                if fluid_params["parent_order_check"]: #Do only want to check parent order here because final check gets volume from formulary names but we want rate from the medication ordered
                    primary_params = fluid_params
                    secondary_params = non_fluid_params
                    rate_source = "fluid"
                    if pd.notna(fluid_params.get("volume")):
                        volume = fluid_params["volume"]
                    elif (pd.notna(fluid_params.get("rate"))) & (pd.notna(fluid_params.get("duration"))) & (fluid_params["rate"] > 0) & (fluid_params["duration"] > 0):
                        volume = fluid_params["rate"] * fluid_params["duration"]
                    print("Using fluid parameters as primary (parent_order_check=True)")
                elif non_fluid_params["final_check"]: #Final check will be true even when parent_order_check is true
                    primary_params = non_fluid_params
                    secondary_params = fluid_params  
                    rate_source = "medication"
                    print("Using medication parameters as primary (final_check=True)")
                
                # Extract volume (prioritize fluid, then medication volume_inf)
                if volume is None: #Enters here only if fluid has parent order but no volume, or if medication is the primary order
                    if pd.notna(fluid_params.get("volume")):
                        volume = fluid_params["volume"]
                        print(f"Using fluid volume: {volume}mL")
                    elif pd.notna(fluid_row.get("volume_inf")):
                        volume = fluid_row["volume_inf"]
                        print(f"Using fluid volume_inf: {volume}mL")
                    elif pd.notna(non_fluid_row.get("volume_inf")):
                        volume = non_fluid_row["volume_inf"]  
                        print(f"Using medication volume_inf: {volume}mL")
                    elif pd.notna(non_fluid_row.get("volume_from_concentration")):
                        volume = non_fluid_row["volume_from_concentration"]
                        print(f"Using medication volume from concentration: {volume}mL")
                    else:
                        print(f"Could not determine volume from orders at med action time: {action_time}")
                
                # Extract rate (from medication unless fluid has parent order)
                rate = None
                if rate_source == "fluid" and pd.notna(primary_params.get("rate")):
                    rate = primary_params["rate"]
                    print(f"Using fluid rate: {rate}")
                elif rate_source == "medication" and pd.notna(primary_params.get("rate")):
                    rate = primary_params["rate"]
                    print(f"Using medication rate: {rate}")
                elif pd.notna(non_fluid_row["rate_default_numeric"]):
                    rate = non_fluid_row["rate_default_numeric"]
                    
                
                # Extract duration
                duration = primary_params.get("duration")
                if pd.isna(duration):
                    duration = secondary_params.get("duration")
                if pd.isna(duration):
                    print("No duration from parameters, imputing to default of 1 hr")
                    duration = 1
                print(f"Using duration: {duration}h")
                
                # Calculate actual duration if this is a Begin Bag or Rate Change
                med_action = fluid_row["med_action"]
                if med_action in ["Begin Bag", "Rate Change"]:
                    # Get all rows for both medications to calculate actual duration
                    all_med_rows = pd.concat([fluid_rows, non_fluid_rows]).sort_values('med_action_time')
                    
                    actual_duration, used_infuse_indices, last_infuse_time = calculate_actual_infusion_duration(
                        fluid_row, all_med_rows, processed_indices
                    )
                    processed_indices.update(used_infuse_indices)
                    
                    if actual_duration > 0:
                        duration = actual_duration
                        med_stop = last_infuse_time
                        print(f"Using actual duration from Infuse actions: {duration:.2f}h")
                    else:
                        med_stop = pd.to_datetime(action_time) + pd.Timedelta(hours=duration)
                        print(f"Using theoretical duration: {duration}h")
                else:
                    med_stop = pd.to_datetime(action_time) + pd.Timedelta(hours=duration)
                

                ## Get infusion periods
                med_start = pd.to_datetime(action_time)
                
                # Handle different medication actions
                if med_action in ["Begin Bag", "Infuse", "Not Recorded"] and not medsdict[order_id][combined_name]["set"]:
                    # First infusion - establish parameters
                    print("Setting initial combined parameters")
                    _initialize_medication_params(medsdict, order_id, combined_name, volume, rate, duration, med_start, med_stop)
                    print(f"INITIAL DILUTION PAIR ({med_action}) - Start={med_start}, Stop={med_stop}, Rate={rate}, Volume={volume}mL, Duration={duration:.2f}h")
                    
                elif med_action == "Rate Change":
                    if not medsdict[order_id][combined_name]["set"]:
                        # First action is Rate Change - treat as initial
                        print("Rate Change as first action - treating as initial")
                        _initialize_medication_params(medsdict, order_id, combined_name, volume, rate, duration, med_start, med_stop)
                        
                    else:
                        # Handle rate change during or after infusion
                        prev_start = medsdict[order_id][combined_name]["med_start"][-1]
                        prev_stop = medsdict[order_id][combined_name]["med_stop"][-1]
                        prev_rate = medsdict[order_id][combined_name]["rate"][-1]
                        prev_volume = medsdict[order_id][combined_name]["volume"][-1]
                        
                        if med_start < prev_stop:
                            # Rate change during active infusion - split period
                            print(f"Rate change during active infusion (was due to end at {prev_stop})")
                            
                            time_elapsed = (med_start - prev_start).total_seconds() / 3600.0
                            volume_delivered = prev_rate * time_elapsed
                            remaining_volume = prev_volume - volume_delivered
                            
                            # Update previous period
                            medsdict[order_id][combined_name]["med_stop"][-1] = med_start
                            medsdict[order_id][combined_name]["duration"][-1] = time_elapsed
                            
                            # Add new period with remaining volume
                            _append_medication_params(medsdict, order_id, combined_name, remaining_volume, rate, duration, med_start, med_stop)
                            
                            print(f"RATE CHANGE (DURING) - Split at {med_start}, remaining volume: {remaining_volume:.1f}mL")
                        else:
                            # Rate change after infusion ended - new period
                            print(f"Rate change after infusion ended")
                            _append_medication_params(medsdict, order_id, combined_name, volume, rate, duration, med_start, med_stop)
                            
                            print(f"RATE CHANGE (AFTER) - New period: Volume={volume}mL, Rate={rate}")
                
                elif med_action in ["Begin Bag", "Infuse", "Not Recorded"] and medsdict[order_id][combined_name]["set"]:
                    # Subsequent infusion
                    print("Adding subsequent infusion period")
                    _append_medication_params(medsdict, order_id, combined_name, volume, rate, duration, med_start, med_stop)
                    
                    print(f"SUBSEQUENT DILUTION PAIR ({med_action}) - Start={med_start}, Stop={med_stop}, Rate={rate}, Volume={volume}mL")
                
                # Mark both rows as processed
                processed_indices.add(fluid_row.name)
                processed_indices.add(non_fluid_row.name)
    
    # Summary logging
    if medsdict[order_id][combined_name]["set"]:
        num_periods = len(medsdict[order_id][combined_name]["med_start"])
        print(f"\n=== DILUTION PAIR SUMMARY: {combined_name} ===")
        print(f"Total infusion periods: {num_periods}")
        for i in range(num_periods):
            start = medsdict[order_id][combined_name]["med_start"][i]
            stop = medsdict[order_id][combined_name]["med_stop"][i]
            rate = medsdict[order_id][combined_name]["rate"][i]
            volume = medsdict[order_id][combined_name]["volume"][i]
            duration = medsdict[order_id][combined_name]["duration"][i]
            print(f"  Period {i+1}: {start} to {stop} | Rate: {rate} | Volume: {volume}mL | Duration: {duration:.2f}h")
        print("==========================================\n")
        
        return medsdict
    else:
        print(f"ERROR: No valid infusion periods established for {combined_name}")
        return None




def process_encounter_new(meds, supertable):
    """
    NEW FUNCTION: Process encounter using multiple meds per order id infusion duration logic.
    """
    # Sort medications chronologically by med_action_time
    meds = meds.sort_values("med_action_time")

    # Filter to infusion medications only (i.e., not injections or syringes)
    imeds = meds.loc[meds["is_infusion"]]
    print(f"Initial infusion meds: {imeds.shape}")

    # Process premix diluents (as these don't give us any information about the infusion)
    imeds = mpu.process_premix(imeds)
    imeds = imeds.loc[imeds.formulary_name != "Not Recorded"]
    print(f"After filtering: {imeds.shape}")
    
    # Get unique order ids for this encounter
    unique_order_ids = imeds["order_med_id"].unique()
    print(f"{len(unique_order_ids)} unique order ids")
    medsdict = {}
    
    for order_id in unique_order_ids:
        print(f'\nProcessing order id: {order_id}')
        
        # Get all rows for this order
        order_rows = imeds.loc[imeds["order_med_id"] == order_id].sort_values('med_action_time')
        unique_meds = order_rows['med_name'].unique()
        
        # Check if this is a dilution pair (exactly 2 medications) 
        # TODO: Add logic to handle more than 2 medications
        if len(unique_meds) == 2:
            print(f"Detected potential dilution pair with medications: {unique_meds}")
            
            fluid_meds = []
            non_fluid_meds = []
            
            # Check if any row for this medication has is_fluid=True
            for med_name in unique_meds:
                med_rows = order_rows[order_rows['med_name'] == med_name]
                if med_rows['is_fluid'].any():
                    fluid_meds.append(med_name)
                else:
                    non_fluid_meds.append(med_name)
            
            if len(fluid_meds) == 1 and len(non_fluid_meds) == 1:
                print(f"Confirmed dilution pair: {non_fluid_meds[0]} (medication) + {fluid_meds[0]} (fluid)")
                
                # Process as dilution pair
                dilution_result = process_dilution_pairs(
                    order_id, fluid_meds[0], non_fluid_meds[0], order_rows, supertable
                )
                
                # Update medsdict with the result of the dilution pair processing
                if dilution_result:
                    medsdict.update(dilution_result)
                    print(f"Successfully processed dilution pair for order {order_id}")
                else:
                    print(f"ERROR: Failed to process dilution pair for order {order_id} - skipping entire order")
                
                continue  # Skip individual processing for this order
            else:
                # Not a valid dilution pair - process individually
                print(f"Not a valid dilution pair (fluid_meds: {len(fluid_meds)}, non_fluid_meds: {len(non_fluid_meds)}) - processing individually")
        
        # Process individually (original logic)
        print(f"Processing {len(unique_meds)} medications individually")
        
        # Initialize medsdict for this order
        medsdict[order_id] = {}
        for med in unique_meds:
            medsdict[order_id][med] = {
                'rate': [], 'duration': [], 'med_start': [], 'med_stop': [], 'volume': [], 'set': False,
                'original_rate': None, 'original_volume': None, 'original_duration': None
            }
        
        all_processed_indices = set()
        
        # Process each medication in this order
        for med_name in unique_meds:
            print(f"\n--- Processing medication: {med_name} ---")
            medsdict, med_processed_indices = process_medication_timeline_new(
                order_id, med_name, order_rows, supertable, medsdict
            )
            all_processed_indices.update(med_processed_indices)
        
        print(f"Processed {len(all_processed_indices)} rows for order {order_id}")
    
    return medsdict


def add_med_volumes_to_final_df(medsdf, formulary_name, med_start, med_stop, rate):
    """
    Add volumes to the final dataframe
    """
    if med_start and med_stop:
        
        # Find the first hourly timestamp that could overlap (the one before or at med_start)
        first_idx = max(0, medsdf.index.get_indexer([med_start], method='ffill')[0])
        if med_start == med_stop:
            medsdf.loc[medsdf.index[first_idx], formulary_name] += rate
        else:
            # Find the last hourly timestamp that could overlap  
            last_idx = medsdf.index.get_indexer([med_stop], method='bfill')[0]
            # Get hourly timestamps that might overlap with med period
            hourly_timestamps = medsdf.index[first_idx:last_idx]
            for timestamp in hourly_timestamps:
                # Each timestamp represents the start of an hour-long period
                period_start = timestamp
                period_end = timestamp + pd.Timedelta(hours=1)
                
                # Calculate overlap between medication period and this hour period
                overlap_start = max(med_start, period_start)
                overlap_end = min(med_stop, period_end)
                
                if overlap_end > overlap_start:
                    overlap_hours = (overlap_end - overlap_start).total_seconds() / 3600.0
                    hourly_volume = rate * overlap_hours
                    medsdf.loc[timestamp, formulary_name] += hourly_volume
    
    return medsdf

def add_to_all_meds_dict(row, all_meds_dict):
    """
    Add meds by med_name to a dictionary that tracks whether a particular med was given or not and when

    Args:
        row: row from the infusion meds (merged with all fluids info dataframe). Should have med_name, med_start
        med_stop, med_class, med_subclass
        all_meds_dict: Dictionary that tracks all meds given for this particular csn
    
    Returns:
        all_meds_dict: With the med from the row added 
    """
    med_name = row['med_name']
    if med_name in all_meds_dict:
        all_meds_dict[med_name]["med_start"].append(row["med_start"])
        all_meds_dict[med_name]["med_stop"].append(row["med_stop"])    
    else:
        all_meds_dict[med_name] = {
            "med_class" : row["med_class"],
            "med_subclass" : row["med_subclass"],
            "med_start" : [row["med_start"]],
            "med_stop" : [row["med_stop"]]
        }
    return all_meds_dict




def make_medsdict_to_dataframe(supertable: pd.DataFrame, medsdict: dict):
    medsdf = pd.DataFrame(index = supertable.index)
    
    meds = []
    for order_id, med_dict in medsdict.items():
        for med, med_data in med_dict.items():
            if med not in meds:
                meds.append(med)
    
    # Create medication columns (volumes) - now supports combined names like "insulin regular + nacl 0.9%"
    for med in meds:
        medsdf[med] = 0.0
    
    # Create error columns 
    for med in meds:
        medsdf[f"{med}_error"] = 0

    for order_id, med_dict in medsdict.items():
        for med, med_data in med_dict.items():
            # Get the lists of parameters for this medication
            rates = med_data.get('rate', [])
            med_starts = med_data.get('med_start', [])
            med_stops = med_data.get('med_stop', [])
            durations = med_data.get('duration', [])
            
            #calculate duration from med_start and med_stop if it is not provided 
            if all(pd.isna(d) for d in durations):
                durations = [med_stops[i] - med_starts[i] for i in range(len(med_starts)) if pd.notna(med_stops[i]) and pd.notna(med_starts[i])]

            has_error = False
            
            # Check if medication has no rate or duration data (final_check was False)
            if not rates or not durations or all(pd.isna(r) for r in rates) or all(pd.isna(d) for d in durations):
                print(f"ERROR: {med} (order {order_id}) has no valid rate/duration data - final_check was False")
                has_error = True
            
            # Check for overlapping periods (error detection)
            if len(med_starts) > 1:
                for i in range(len(med_starts) - 1):
                    if (i < len(med_stops) and i+1 < len(med_starts) and 
                        pd.notna(med_stops[i]) and pd.notna(med_starts[i+1])):
                        if med_starts[i+1] < med_stops[i]:
                            print(f"ERROR: Overlapping infusion periods detected for {med} (order {order_id})")
                            print(f"  Period {i}: {med_starts[i]} to {med_stops[i]}")
                            print(f"  Period {i+1}: {med_starts[i+1]} to {med_stops[i+1]}")
                            has_error = True
            
            # Set error flag if any errors detected
            if has_error:
                medsdf[f"{med}_error"] = 1
            
            # Process each infusion period (only if no errors)
            if not has_error:
                for i in range(len(med_starts)):
                    if (i < len(rates) and i < len(med_stops) and 
                        pd.notna(rates[i]) and pd.notna(med_starts[i]) and pd.notna(med_stops[i])):
                        
                        rate = rates[i]
                        med_start = pd.to_datetime(med_starts[i])
                        med_stop = pd.to_datetime(med_stops[i])
                        
                        # Skip if rate is 0 or negative, or if start >= stop
                        if rate <= 0 or med_start >= med_stop:
                            print(f"ERROR: Invalid period {i} for {med}: rate={rate}, start={med_start}, stop={med_stop}")
                            medsdf[f"{med}_error"] = 1
                            continue
                        
                        # Distribute volume hourly based on rate using existing function
                        medsdf = add_med_volumes_to_final_df(medsdf, med, med_start, med_stop, rate)
                    else:
                        print(f"ERROR: Missing data for period {i} for {med}: rate={rates[i] if i < len(rates) else 'N/A'}")
                        medsdf[f"{med}_error"] = 1

    return medsdf


def main():
    supertable_path = Path("/labs/collab/K-lab-MODS/MODS-PHI/Encounter_Pickles/emHolder_OutlierCorrected/2019_csvs")
    allmeds = pd.read_csv("/labs/collab/K-lab-MODS/MODS-PHI/Emory_Data/2019/FLUIDS_2019.dsv", sep = "|")
    meds_mapping = pd.read_csv("../groupings/em_infusion_meds_classification_final.csv")

    allmeds["volume_inf"] = allmeds["formulary_name"].map(dict(zip(meds_mapping["formulary_name"], meds_mapping["volume_numeric"])))
    allmeds["volume_inf_unit"] = allmeds["formulary_name"].map(dict(zip(meds_mapping["formulary_name"], meds_mapping["volume_unit"])))

    allmeds["amount_inf"] = allmeds["formulary_name"].map(dict(zip(meds_mapping["formulary_name"], meds_mapping["amount_numeric"])))
    allmeds["amount_inf_unit"] = allmeds["formulary_name"].map(dict(zip(meds_mapping["formulary_name"], meds_mapping["amount_unit"])))

    allmeds["is_anesthesia"] = allmeds["formulary_name"].map(dict(zip(meds_mapping["formulary_name"], meds_mapping["is_anesthesia"])))
    allmeds["is_infusion"] = allmeds["formulary_name"].map(dict(zip(meds_mapping["formulary_name"], meds_mapping["is_infusion"])))
    allmeds["is_fluid"] = allmeds["formulary_name"].map(dict(zip(meds_mapping["formulary_name"], meds_mapping["is_fluids"])))

    encounters_list = list(supertable_path.glob("*.csv"))
    print(len(encounters_list))

    encounters = pd.read_csv("/labs/collab/K-lab-MODS/MODS-PHI/Emory_Data/2019/CJSEPSIS_BEDLOCATION_2019.dsv", sep = "|")
    icu = encounters.loc[encounters.bed_unit.str.contains("ICU")]
    encounters_list_icu = icu["csn"].values
    
    idx = 80

    supertable = pd.read_csv(supertable_path / f"{encounters_list_icu[idx]}.csv")
    supertable_index = supertable["Unnamed: 0"]
    supertable_index = pd.to_datetime(supertable_index)
    supertable.daily_weight_kg = supertable["daily_weight_kg"].ffill().bfill()
    supertable = supertable.set_index(supertable_index)

    meds = allmeds.loc[allmeds.csn == encounters_list_icu[idx]] 
    
    # Choose which processing method to use
    use_new_logic = True  # Set to False to use original logic
    
    if use_new_logic:
        print("=== Using NEW infusion duration logic ===")
        medsdict = process_encounter_new(meds, supertable)
    else:
        print("=== Using ORIGINAL logic ===")
        medsdict = process_encounter(meds, supertable) 



if __name__ == "__main__":
    main()

