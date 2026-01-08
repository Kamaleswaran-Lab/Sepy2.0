from pathlib import Path
import pandas as pd
import numpy as np
import sys
sys.path.append("../")
import data_preprocess.medication_processor as mp


def identify_med_formulary_pairs(order_rows):
    """
    Identify unique (med_name, formulary_name) pairs from order rows.
    
    Args:
        order_rows: DataFrame with all rows for a given order_id
        
    Returns:
        dict: {(med_name, formulary_name): DataFrame of rows for that pair}
    """
    pairs = {}
    
    for (med_name, formulary_name), group in order_rows.groupby(['med_name', 'formulary_name']):
        pairs[(med_name, formulary_name)] = group
    
    return pairs


def filter_not_recorded_meds(order_rows):
    """
    Remove med_names where ALL rows have formulary_name='Not Recorded'.
    This indicates the formulary name couldn't be imputed within this order.
    
    Args:
        order_rows: DataFrame with all rows for a given order_id
        
    Returns:
        tuple: (filtered_rows, removed_med_names)
    """
    removed_meds = []
    
    for med_name in order_rows['med_name'].unique():
        med_rows = order_rows[order_rows['med_name'] == med_name]
        if (med_rows['formulary_name'] == 'Not Recorded').all():
            removed_meds.append(med_name)
            print(f"Removing {med_name} - all rows are 'Not Recorded'")
    
    filtered_rows = order_rows[~order_rows['med_name'].isin(removed_meds)]
    return filtered_rows, removed_meds


def classify_all_pairs(pairs_dict):
    """
    Classify all (med_name, formulary_name) pairs in an order.
    
    For each pair:
    - Determines if it's a fluid from is_fluid column
    - For non-fluids: checks if it's an infusion by verifying all med_action 
      values are in {'Begin Bag', 'Rate Change', 'Infuse'}
    
    Args:
        pairs_dict: Dictionary mapping (med_name, formulary_name) -> DataFrame of rows
        
    Returns:
        dict: {(med_name, formulary_name): {'is_fluid': bool, 'is_infusion': bool}}
    """
    classifications = {}
    
    for pair_key, pair_rows in pairs_dict.items():
        med_name, formulary_name = pair_key
        
        # Check is_fluid from the data
        is_fluid_values = pair_rows['is_fluid'].dropna().unique()
        if len(is_fluid_values) > 1:
            print(f"WARNING: {med_name}||{formulary_name} has inconsistent is_fluid values: {is_fluid_values}")
            is_fluid = is_fluid_values[0]  # Use first value
        elif len(is_fluid_values) == 1:
            is_fluid = is_fluid_values[0]
        else:
            is_fluid = False  # Default if all NaN
        
        # Determine is_infusion
        if is_fluid:
            # Fluids are always infusions
            is_infusion = True
        else:
            # For non-fluids, check if all actions are infusion-related
            infusion_actions = {'Begin Bag', 'Rate Change', 'Infuse'}
            all_actions = set(pair_rows['med_action'].unique())
            is_infusion = all_actions.issubset(infusion_actions)
        
        classifications[pair_key] = {
            'is_fluid': is_fluid,
            'is_infusion': is_infusion
        }
        
        print(f"  {med_name}||{formulary_name}: is_fluid={is_fluid}, is_infusion={is_infusion}")
    
    return classifications


def get_common_and_noncommon_timestamps(fluid_rows, med_rows):
    """
    Get common and non-common timestamps between fluid and medication rows.
    
    Args:
        fluid_rows: DataFrame for fluid medication
        med_rows: DataFrame for non-fluid medication
        
    Returns:
        tuple: (common_times, fluid_only_times, med_only_times)
    """
    fluid_times = set(pd.to_datetime(fluid_rows['med_action_time']))
    med_times = set(pd.to_datetime(med_rows['med_action_time']))
    
    common_times = fluid_times.intersection(med_times)
    fluid_only_times = fluid_times.difference(common_times)
    med_only_times = med_times.difference(common_times)
    
    return sorted(common_times), sorted(fluid_only_times), sorted(med_only_times)


def process_dilution_pair_common_timestamp(
    action_time,
    fluid_row,
    med_row,
    order_id,
    pair_key,
    medsdict,
    supertable,
    all_med_rows,
    processed_indices
):
    """
    Process a single common timestamp for a dilution pair.
    
    Args:
        action_time: The common timestamp
        fluid_row: Row from fluid medication
        med_row: Row from non-fluid medication
        order_id: Order ID
        pair_key: Tuple key for this dilution pair in medsdict
        medsdict: Medication dictionary
        supertable: Patient data table
        all_med_rows: All rows for the medication (for calculating actual duration)
        processed_indices: Set of processed row indices
        
    Returns:
        tuple: (updated_medsdict, updated_processed_indices)
    """
    med_action = fluid_row['med_action']
    print(f"\nProcessing common timestamp {action_time} - Action: {med_action}")
    
    # Get parameters from both sources
    fluid_params = mp.get_order_params_for_row(fluid_row, supertable)
    med_params = mp.get_order_params_for_row(med_row, supertable)
    
    # Extract volume (prioritize fluid)
    volume = None
    if pd.notna(fluid_params.get("volume")):
        volume = fluid_params["volume"]
        print(f"Using fluid volume: {volume}mL")
    elif pd.notna(fluid_row.get("volume_inf")):
        volume = fluid_row["volume_inf"]
        print(f"Using fluid volume_inf: {volume}mL")
    elif pd.notna(med_row.get("volume_inf")):
        volume = med_row["volume_inf"]
        print(f"Using med volume_inf: {volume}mL")
    elif pd.notna(med_row.get("volume_from_concentration")):
        volume = med_row["volume_from_concentration"]
        print(f"Using med volume_from_concentration: {volume}mL")
    
    # Extract rate (preference: fluid parent order -> med params -> defaults)
    rate = None
    rate_source = None
    
    # Check fluid parent order first
    if fluid_params["parent_order_check"] and pd.notna(fluid_params.get("rate")) and fluid_params["rate"] != 0:
        rate = fluid_params["rate"]
        rate_source = "fluid_parent_order"
        print(f"Using fluid parent order rate: {rate}mL/h")
    
    # Check med params
    elif med_params["final_check"] and pd.notna(med_params.get("rate")):
        med_rate = med_params["rate"]
        
        # If we had a fluid rate, check for exact match
        if rate_source == "fluid_parent_order":
            if med_rate == rate:
                rate_source = "med_params"
                print(f"Med rate matches fluid rate: {rate}mL/h")
            else:
                # Check if we have previous infusion with original params
                if medsdict[order_id][pair_key]["set"]:
                    original_rate = medsdict[order_id][pair_key]["original_rate"]
                    rate = original_rate
                    rate_source = "original_params"
                    print(f"Med rate {med_rate} ≠ fluid rate {fluid_params['rate']}, using original rate: {rate}mL/h")
                else:
                    raise ValueError(f"Med rate {med_rate} ≠ fluid rate {rate}, no previous infusion to reference")
        else:
            rate = med_rate
            rate_source = "med_params"
            print(f"Using med rate: {rate}mL/h")
    
    # Fallback to defaults if no rate found
    if rate is None:
        if medsdict[order_id][pair_key]["set"]:
            rate = medsdict[order_id][pair_key]["original_rate"]
            rate_source = "original_params"
            print(f"Using original rate: {rate}mL/h")
        elif pd.notna(med_row.get("rate_default_numeric")):
            rate = med_row["rate_default_numeric"]
            rate_source = "med_default"
            print(f"Using med rate_default_numeric: {rate}mL/h")
        else:
            raise ValueError(f"No rate found for dilution pair at {action_time}")
    
    # Extract duration based on rate source
    duration = None
    if rate_source == "fluid_parent_order":
        duration = fluid_params.get("duration") if pd.notna(fluid_params.get("duration")) else None 
    elif rate_source == "med_params": 
        duration = med_params.get("duration") if pd.notna(med_params.get("duration")) else None
    elif rate_source == "original_params":
        duration = medsdict[order_id][pair_key]["original_duration"]

    if rate_source == "med_default" or pd.isna(duration):
        duration = 1.0  # Default
        print(f"No duration from params, defaulting to 1 hour")
    else:
        print(f"Using duration: {duration}h")
    
    med_start = pd.to_datetime(action_time)
    
    # Handle different medication actions
    if med_action == "Begin Bag":
        # Calculate actual duration from Infuse actions
        actual_duration, used_infuse_indices, last_infuse_time = mp.calculate_actual_infusion_duration(
            med_row, all_med_rows, processed_indices
        )
        processed_indices.update(used_infuse_indices)
        
        if actual_duration > 0:
            duration = actual_duration
            med_stop = last_infuse_time
            print(f"Using actual duration from med Infuse actions: {duration:.2f}h")
        else:
            med_stop = med_start + pd.Timedelta(hours=duration)
            print(f"Using theoretical duration: {duration}h")
        
        # Recalculate rate if needed
        if pd.notna(volume) and duration > 0:
            rate = volume / duration
            print(f"Recalculated rate from volume and duration: {rate:.2f}mL/h")
        
        if not medsdict[order_id][pair_key]["set"]:
            # First infusion
            mp._initialize_medication_params(medsdict, order_id, pair_key, volume, rate, duration, med_start, med_stop)
            print(f"DILUTION PAIR BEGIN BAG (INITIAL) - Start={med_start}, Stop={med_stop}, Rate={rate:.1f}mL/h, Volume={volume:.1f}mL")
        else:
            # Subsequent bag
            mp._append_medication_params(medsdict, order_id, pair_key, volume, rate, duration, med_start, med_stop)
            print(f"DILUTION PAIR BEGIN BAG (SUBSEQUENT) - Start={med_start}, Stop={med_stop}, Rate={rate:.1f}mL/h, Volume={volume:.1f}mL")
    
    elif med_action == "Rate Change":
        if not medsdict[order_id][pair_key]["set"]:
            # First action is rate change - treat as initial bag
            # Calculate actual duration
            actual_duration, used_infuse_indices, last_infuse_time = mp.calculate_actual_infusion_duration(
                med_row, all_med_rows, processed_indices
            )
            processed_indices.update(used_infuse_indices)
            
            if actual_duration > 0:
                duration = actual_duration
                med_stop = last_infuse_time
                print(f"Using actual duration from med Infuse actions: {duration:.2f}h")
            else:
                med_stop = med_start + pd.Timedelta(hours=duration)
                print(f"Using theoretical duration: {duration}h")
            
            # Recalculate rate if needed
            if pd.notna(volume) and duration > 0:
                rate = volume / duration
            
            mp._initialize_medication_params(medsdict, order_id, pair_key, volume, rate, duration, med_start, med_stop)
            print(f"DILUTION PAIR RATE CHANGE (INITIAL) - Start={med_start}, Stop={med_stop}, Rate={rate:.1f}mL/h, Volume={volume:.1f}mL")
        else:
            # Subsequent rate change - check if during active infusion or after
            prev_start = medsdict[order_id][pair_key]["med_start"][-1]
            prev_stop = medsdict[order_id][pair_key]["med_stop"][-1]
            prev_rate = medsdict[order_id][pair_key]["rate"][-1]
            prev_volume = medsdict[order_id][pair_key]["volume"][-1]
            
            # Check if rate actually changed
            if pd.notna(rate) and pd.notna(prev_rate) and rate == prev_rate:
                print(f"Rate change but no actual rate difference ({rate}mL/h) - ignoring")
                # Still mark rows as processed
                processed_indices.add(fluid_row.name)
                processed_indices.add(med_row.name)
                return medsdict, processed_indices
            
            # Validate new rate
            if pd.isna(rate) or rate < 0:
                print(f"Invalid new rate ({rate}) - ignoring rate change")
                processed_indices.add(fluid_row.name)
                processed_indices.add(med_row.name)
                return medsdict, processed_indices
            
            if med_start < prev_stop:
                # Rate change during active infusion - split the period
                print(f"Rate change during active infusion (was due to end at {prev_stop})")
                
                # Calculate what was already delivered
                time_elapsed = (med_start - prev_start).total_seconds() / 3600.0
                volume_delivered = prev_rate * time_elapsed
                remaining_volume = prev_volume - volume_delivered
                
                print(f"Time elapsed: {time_elapsed:.2f}h, Volume delivered: {volume_delivered:.1f}mL, Remaining: {remaining_volume:.1f}mL")
                
                # Update previous period to end at rate change time
                medsdict[order_id][pair_key]["med_stop"][-1] = med_start
                medsdict[order_id][pair_key]["duration"][-1] = time_elapsed
                
                # Calculate actual duration for new period
                actual_duration, used_infuse_indices, last_infuse_time = mp.calculate_actual_infusion_duration(
                    med_row, all_med_rows, processed_indices
                )
                processed_indices.update(used_infuse_indices)
                
                # Determine final duration for new period
                if rate == 0:
                    # Medication stopped
                    if actual_duration > 0:
                        final_duration = actual_duration
                        new_stop_time = last_infuse_time
                        print(f"Medication stopped - using actual duration: {final_duration:.2f}h")
                    else:
                        final_duration = 0
                        new_stop_time = med_start
                        print(f"Medication stopped immediately at {med_start}")
                elif actual_duration > 0:
                    final_duration = actual_duration
                    new_stop_time = last_infuse_time
                    print(f"Using actual duration from Infuse actions: {final_duration:.2f}h")
                else:
                    # No infuse actions - calculate from remaining volume and new rate
                    if remaining_volume > 0 and rate > 0:
                        final_duration = remaining_volume / rate
                        new_stop_time = med_start + pd.Timedelta(hours=final_duration)
                        print(f"No Infuse actions - using theoretical duration: {final_duration:.2f}h")
                    else:
                        print(f"Cannot calculate duration - skipping rate change")
                        processed_indices.add(fluid_row.name)
                        processed_indices.add(med_row.name)
                        return medsdict, processed_indices
                
                # Add new period with remaining volume and new rate
                mp._append_medication_params(medsdict, order_id, pair_key, remaining_volume, rate, final_duration, med_start, new_stop_time)
                print(f"DILUTION PAIR RATE CHANGE (DURING) - Split at {med_start}, Rate={rate:.1f}mL/h, Volume={remaining_volume:.1f}mL, Duration={final_duration:.2f}h")
            
            else:
                # Rate change after previous infusion ended - new bag
                print(f"Rate change after infusion ended (ended at {prev_stop}) - starting new period")
                
                # Calculate actual duration
                actual_duration, used_infuse_indices, last_infuse_time = mp.calculate_actual_infusion_duration(
                    med_row, all_med_rows, processed_indices
                )
                processed_indices.update(used_infuse_indices)
                
                if actual_duration > 0:
                    duration = actual_duration
                    med_stop = last_infuse_time
                    print(f"Using actual duration from med Infuse actions: {duration:.2f}h")
                else:
                    med_stop = med_start + pd.Timedelta(hours=duration)
                    print(f"Using theoretical duration: {duration}h")
                
                # Recalculate rate if needed
                if pd.notna(volume) and duration > 0:
                    rate = volume / duration
                
                mp._append_medication_params(medsdict, order_id, pair_key, volume, rate, duration, med_start, med_stop)
                print(f"DILUTION PAIR RATE CHANGE (AFTER) - New period: Rate={rate:.1f}mL/h, Volume={volume:.1f}mL, Duration={duration:.2f}h")
    
    elif med_action == "Infuse":
        # Infuse action - similar to Begin Bag
        # Calculate actual duration
        actual_duration, used_infuse_indices, last_infuse_time = mp.calculate_actual_infusion_duration(
            med_row, all_med_rows, processed_indices
        )
        processed_indices.update(used_infuse_indices)
        
        if actual_duration > 0:
            duration = actual_duration
            med_stop = last_infuse_time
            print(f"Using actual duration from med Infuse actions: {duration:.2f}h")
        else:
            med_stop = med_start + pd.Timedelta(hours=duration)
            print(f"Using theoretical duration: {duration}h")
        
        # Recalculate rate if needed
        if pd.notna(volume) and duration > 0:
            rate = volume / duration
        
        if not medsdict[order_id][pair_key]["set"]:
            mp._initialize_medication_params(medsdict, order_id, pair_key, volume, rate, duration, med_start, med_stop)
            print(f"DILUTION PAIR INFUSE (INITIAL) - Start={med_start}, Stop={med_stop}, Rate={rate:.1f}mL/h, Volume={volume:.1f}mL")
        else:
            mp._append_medication_params(medsdict, order_id, pair_key, volume, rate, duration, med_start, med_stop)
            print(f"DILUTION PAIR INFUSE (SUBSEQUENT) - Start={med_start}, Stop={med_stop}, Rate={rate:.1f}mL/h, Volume={volume:.1f}mL")
    
    elif med_action == "Not Recorded":
        # Not Recorded should have explicit med_stop in the data
        if pd.notna(med_row.get("med_stop")):
            med_stop = pd.to_datetime(med_row["med_stop"])
            duration = (med_stop - med_start).total_seconds() / 3600.0
            print(f"Not Recorded with explicit times: start={med_start}, stop={med_stop}, duration={duration:.2f}h")
            
            # Recalculate rate or volume if needed
            if pd.notna(volume) and pd.notna(rate):
                # Both exist, recalculate rate from actual duration
                rate = volume / duration if duration > 0 else rate
            elif pd.notna(volume):
                # Have volume, calculate rate
                rate = volume / duration if duration > 0 else None
            elif pd.notna(rate):
                # Have rate, calculate volume
                volume = rate * duration
        else:
            # No explicit stop time, use theoretical duration
            print(f"Not Recorded without explicit stop time, using theoretical duration: {duration}h")
            med_stop = med_start + pd.Timedelta(hours=duration)
        
        # Recalculate rate if we have volume and duration
        if pd.notna(volume) and duration > 0:
            rate = volume / duration
        
        if not medsdict[order_id][pair_key]["set"]:
            mp._initialize_medication_params(medsdict, order_id, pair_key, volume, rate, duration, med_start, med_stop)
            print(f"DILUTION PAIR NOT RECORDED (INITIAL) - Start={med_start}, Stop={med_stop}, Rate={rate:.1f}mL/h, Volume={volume:.1f}mL")
        else:
            mp._append_medication_params(medsdict, order_id, pair_key, volume, rate, duration, med_start, med_stop)
            print(f"DILUTION PAIR NOT RECORDED (SUBSEQUENT) - Start={med_start}, Stop={med_stop}, Rate={rate:.1f}mL/h, Volume={volume:.1f}mL")
    
    else:
        # Unknown action type
        print(f"WARNING: Unknown action type '{med_action}' - skipping")
        processed_indices.add(fluid_row.name)
        processed_indices.add(med_row.name)
        return medsdict, processed_indices
    
    # Mark both rows as processed
    processed_indices.add(fluid_row.name)
    processed_indices.add(med_row.name)
    
    return medsdict, processed_indices


def process_dilution_pair_extended(
    order_id,
    fluid_pair_key,
    med_pair_key,
    order_rows,
    supertable
):
    """
    Process dilution pairs with separate handling for common and non-common timestamps.
    
    Args:
        order_id: Order ID
        fluid_pair_key: (med_name, formulary_name) for fluid
        med_pair_key: (med_name, formulary_name) for medication
        order_rows: All rows for this order_id
        supertable: Patient data table
        
    Returns:
        dict: medsdict for this order
    """
    print(f"\nProcessing extended dilution pair: {med_pair_key} + {fluid_pair_key}")
    
    # Get rows for each pair
    fluid_rows = order_rows[
        (order_rows['med_name'] == fluid_pair_key[0]) & 
        (order_rows['formulary_name'] == fluid_pair_key[1])
    ].sort_values('med_action_time')
    
    med_rows = order_rows[
        (order_rows['med_name'] == med_pair_key[0]) & 
        (order_rows['formulary_name'] == med_pair_key[1])
    ].sort_values('med_action_time')
    
    # Create combined pair key for storage
    combined_key = (med_pair_key[0], med_pair_key[1], fluid_pair_key[0], fluid_pair_key[1])
    
    # Initialize medsdict
    medsdict = {
        order_id: {
            combined_key: {
                'rate': [], 'duration': [], 'med_start': [], 'med_stop': [], 'volume': [], 'set': False,
                'original_rate': None, 'original_volume': None, 'original_duration': None
            }
        }
    }
    
    processed_indices = set()
    
    # Get common and non-common timestamps
    common_times, fluid_only_times, med_only_times = get_common_and_noncommon_timestamps(fluid_rows, med_rows)
    
    print(f"Found {len(common_times)} common timestamps, {len(fluid_only_times)} fluid-only, {len(med_only_times)} med-only")
    
    # Process common timestamps
    for action_time in common_times:
        fluid_time_rows = fluid_rows[pd.to_datetime(fluid_rows['med_action_time']) == action_time]
        med_time_rows = med_rows[pd.to_datetime(med_rows['med_action_time']) == action_time]
        
        for _, fluid_row in fluid_time_rows.iterrows():
            for _, med_row in med_time_rows.iterrows():
                if fluid_row.name in processed_indices or med_row.name in processed_indices:
                    continue
                
                try:
                    medsdict, processed_indices = process_dilution_pair_common_timestamp(
                        action_time, fluid_row, med_row, order_id, combined_key,
                        medsdict, supertable, med_rows, processed_indices
                    )
                except Exception as e:
                    print(f"ERROR processing common timestamp {action_time}: {e}")
                    raise
    
    # Process non-common timestamps - try as single infusions
    errors = []
    
    # Process fluid-only timestamps
    if fluid_only_times:
        print(f"\nProcessing {len(fluid_only_times)} fluid-only timestamps")
        for action_time in fluid_only_times:
            fluid_time_rows = fluid_rows[pd.to_datetime(fluid_rows['med_action_time']) == action_time]
            for _, fluid_row in fluid_time_rows.iterrows():
                if fluid_row.name in processed_indices:
                    continue
                
                try:
                    # Try to process as single infusion
                    # Create temporary medsdict entry for this fluid
                    temp_key = fluid_pair_key
                    if temp_key not in medsdict[order_id]:
                        medsdict[order_id][temp_key] = {
                            'rate': [], 'duration': [], 'med_start': [], 'med_stop': [], 'volume': [], 'set': False,
                            'original_rate': None, 'original_volume': None, 'original_duration': None
                        }
                    
                    # Process using single medication logic
                    print(f"Attempting to process fluid-only timestamp {action_time} as single infusion")
                    # This would require adapting process_medication_timeline_new for single rows
                    # For now, raise error as specified
                    raise NotImplementedError(f"Cannot process fluid-only timestamp {action_time} - not yet implemented")
                    
                except Exception as e:
                    error_msg = f"ERROR processing fluid-only timestamp {action_time}: {e}"
                    print(error_msg)
                    errors.append(error_msg)
    
    # Process med-only timestamps
    if med_only_times:
        print(f"\nProcessing {len(med_only_times)} med-only timestamps")
        for action_time in med_only_times:
            med_time_rows = med_rows[pd.to_datetime(med_rows['med_action_time']) == action_time]
            for _, med_row in med_time_rows.iterrows():
                if med_row.name in processed_indices:
                    continue
                
                try:
                    # Try to process as single infusion
                    temp_key = med_pair_key
                    if temp_key not in medsdict[order_id]:
                        medsdict[order_id][temp_key] = {
                            'rate': [], 'duration': [], 'med_start': [], 'med_stop': [], 'volume': [], 'set': False,
                            'original_rate': None, 'original_volume': None, 'original_duration': None
                        }
                    
                    print(f"Attempting to process med-only timestamp {action_time} as single infusion")
                    raise NotImplementedError(f"Cannot process med-only timestamp {action_time} - not yet implemented")
                    
                except Exception as e:
                    error_msg = f"ERROR processing med-only timestamp {action_time}: {e}"
                    print(error_msg)
                    errors.append(error_msg)
    
    if errors:
        print(f"\n=== ERRORS in non-common timestamp processing ===")
        for error in errors:
            print(f"  {error}")
    
    # Summary
    if medsdict[order_id][combined_key]["set"]:
        num_periods = len(medsdict[order_id][combined_key]["med_start"])
        print(f"\n=== DILUTION PAIR SUMMARY: {combined_key} ===")
        print(f"Total infusion periods: {num_periods}")
        for i in range(num_periods):
            start = medsdict[order_id][combined_key]["med_start"][i]
            stop = medsdict[order_id][combined_key]["med_stop"][i]
            rate = medsdict[order_id][combined_key]["rate"][i]
            volume = medsdict[order_id][combined_key]["volume"][i]
            duration = medsdict[order_id][combined_key]["duration"][i]
            print(f"  Period {i+1}: {start} to {stop} | Rate: {rate:.1f}mL/h | Volume: {volume:.1f}mL | Duration: {duration:.2f}h")
        print("=" * 80)
        
        return medsdict
    else:
        print(f"ERROR: No valid infusion periods established for dilution pair")
        return None


def process_order_multi_med(order_id, order_rows, supertable, all_meds_dict):
    """
    Process an order with multiple medications, tracking by (med_name, formulary_name) pairs.
    
    Args:
        order_id: Order ID to process
        order_rows: All rows for this order_id
        supertable: Patient data table
        all_meds_dict: Dictionary for non-infusion medications
        
    Returns:
        tuple: (medsdict, updated_all_meds_dict)
    """
    print(f"\n{'='*80}")
    print(f"Processing order {order_id} with multiple medications")
    print(f"{'='*80}")
    
    # Step 1: Filter out meds where all rows are "Not Recorded"
    filtered_rows, removed_meds = filter_not_recorded_meds(order_rows)
    
    if len(filtered_rows) == 0:
        print("All meds removed due to 'Not Recorded' - skipping order")
        return {}, all_meds_dict
    
    # Step 2: Identify (med_name, formulary_name) pairs
    pairs = identify_med_formulary_pairs(filtered_rows)
    print(f"\nFound {len(pairs)} unique (med_name, formulary_name) pairs:")
    for pair_key in pairs.keys():
        print(f"  {pair_key[0]} || {pair_key[1]} ({len(pairs[pair_key])} rows)")
    
    # Step 3: Classify all pairs together
    print(f"\nClassifying pairs:")
    pair_classifications = classify_all_pairs(pairs)
    
    # Step 4: Separate non-infusion pairs
    non_infusion_pairs = [(k, v) for k, v in pair_classifications.items() if not v['is_infusion']]
    infusion_pairs = [(k, v) for k, v in pair_classifications.items() if v['is_infusion']]
    
    # Process non-infusion pairs
    for pair_key, _ in non_infusion_pairs:
        print(f"\nProcessing non-infusion pair: {pair_key[0]} || {pair_key[1]}")
        pair_rows = pairs[pair_key]
        for idx, row in pair_rows.iterrows():
            all_meds_dict = mp.add_to_all_meds_dict(row, all_meds_dict)
        print(f"Added to all_meds_dict")
    
    # Filter to infusion pairs only
    remaining_pairs = {k: pairs[k] for k, _ in infusion_pairs}
    
    if len(remaining_pairs) == 0:
        print("\nNo infusion pairs to process")
        return {}, all_meds_dict
    
    # Check if all remaining rows are "Not Recorded"
    all_remaining_rows = pd.concat([df for df in remaining_pairs.values()])
    if (all_remaining_rows['formulary_name'] == 'Not Recorded').all():
        print("\nAll remaining infusion rows are 'Not Recorded' - skipping (likely small diluents)")
        return {}, all_meds_dict
    
    # Process based on number of infusion pairs
    print(f"\n{len(remaining_pairs)} infusion pairs to process")
    
    if len(remaining_pairs) == 1:
        # Single infusion pair - process normally
        pair_key = list(remaining_pairs.keys())[0]
        print(f"\nProcessing single infusion pair: {pair_key[0]} || {pair_key[1]}")
        
        medsdict = {order_id: {}}
        medsdict[order_id][pair_key] = {
            'rate': [], 'duration': [], 'med_start': [], 'med_stop': [], 'volume': [], 'set': False,
            'original_rate': None, 'original_volume': None, 'original_duration': None
        }
        
        # Use existing single medication processing logic
        medsdict, _ = mp.process_medication_timeline_new(
            order_id, pair_key[0], all_remaining_rows, supertable, medsdict
        )
        
        return medsdict, all_meds_dict
    
    elif len(remaining_pairs) == 2:
        # Potential dilution pair
        pair_keys = list(remaining_pairs.keys())
        pair_key_1, pair_key_2 = pair_keys[0], pair_keys[1]
        
        is_fluid_1 = pair_classifications[pair_key_1]['is_fluid']
        is_fluid_2 = pair_classifications[pair_key_2]['is_fluid']
        
        # Check if one is fluid and one is not
        if is_fluid_1 and not is_fluid_2:
            fluid_pair = pair_key_1
            med_pair = pair_key_2
        elif is_fluid_2 and not is_fluid_1:
            fluid_pair = pair_key_2
            med_pair = pair_key_1
        else:
            raise ValueError(f"Dilution pair must have exactly one fluid - found is_fluid_1={is_fluid_1}, is_fluid_2={is_fluid_2}")
        
        print(f"\nConfirmed dilution pair:")
        print(f"  Medication: {med_pair[0]} || {med_pair[1]}")
        print(f"  Fluid: {fluid_pair[0]} || {fluid_pair[1]}")
        
        # Process dilution pair with extended logic
        medsdict = process_dilution_pair_extended(
            order_id, fluid_pair, med_pair, all_remaining_rows, supertable
        )
        
        if medsdict is None:
            print(f"ERROR: Failed to process dilution pair")
            return {}, all_meds_dict
        
        return medsdict, all_meds_dict
    
    else:
        # More than 2 infusion pairs
        raise NotImplementedError(f"Cannot handle {len(remaining_pairs)} infusion pairs in one order")


def process_encounter_multi_med(meds, supertable):
    """
    Process encounter with support for multiple medications per order_id.
    
    This function orchestrates the processing of all medications in an encounter,
    properly handling orders with multiple medications by tracking (med_name, formulary_name) pairs.
    
    Args:
        meds: DataFrame with all medication data for the encounter
        supertable: Patient data table with daily_weight_kg
        
    Returns:
        tuple: (medsdict, all_meds_dict)
    """
    # Sort medications chronologically
    meds = meds.sort_values("med_action_time")
    
    # Filter to infusion medications only
    imeds = meds.loc[meds["is_infusion"]]
    print(f"Initial infusion meds: {imeds.shape}")
    
    # Process premix diluents
    imeds = mp.process_premix(imeds)
    imeds = imeds.loc[imeds.formulary_name != "Not Recorded"]
    print(f"After filtering: {imeds.shape}")
    
    # Get unique order ids
    unique_order_ids = imeds["order_med_id"].unique()
    print(f"{len(unique_order_ids)} unique order ids")
    
    medsdict = {}
    all_meds_dict = {}
    
    for order_id in unique_order_ids:
        print(f'\n{"="*80}')
        print(f'Processing order id: {order_id}')
        print(f'{"="*80}')
        
        order_rows = imeds.loc[imeds["order_med_id"] == order_id].sort_values('med_action_time')
        unique_meds = order_rows['med_name'].unique()
        
        if len(unique_meds) == 1:
            # Single medication - use original logic
            print(f"Single medication: {unique_meds[0]}")
            
            medsdict[order_id] = {}
            med_name = unique_meds[0]
            medsdict[order_id][med_name] = {
                'rate': [], 'duration': [], 'med_start': [], 'med_stop': [], 'volume': [], 'set': False,
                'original_rate': None, 'original_volume': None, 'original_duration': None
            }
            
            medsdict, _ = mp.process_medication_timeline_new(
                order_id, med_name, order_rows, supertable, medsdict
            )
        
        else:
            # Multiple medications - use new logic
            print(f"Multiple medications ({len(unique_meds)}): {list(unique_meds)}")
            
            order_medsdict, all_meds_dict = process_order_multi_med(
                order_id, order_rows, supertable, all_meds_dict
            )
            
            if order_medsdict:
                medsdict.update(order_medsdict)
    
    return medsdict, all_meds_dict


