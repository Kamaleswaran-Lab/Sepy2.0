from pathlib import Path
import pandas as pd
import numpy as np
import sys
sys.path.append("../")
import tqdm

import re
from typing import Optional, Dict
import med_processing_utils as mpu

def get_order_params_for_dilution_pairs(fluid_row, med_row, supertable, verbose = False):
    # will return parent_order_check False if the FLUID doesnt have a parent order 
    # will return final_check False if the med and fluid rate dont match (if fluid has a parent order ) or cant find rate at all anywhere
    med_action = fluid_row['med_action']
    action_time = fluid_row['med_action_time']
    print(f"\nProcessing common timestamp {action_time} - Action: {med_action}")
    
    # Get parameters from both sources
    fluid_params = get_order_params_for_row(fluid_row, supertable)
    med_params = get_order_params_for_row(med_row, supertable)
    
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
    else:
        print("ERROR: Cannot get volume from fluid row")
        return {
                "volume": None,
                "rate": None,
                "duration": None,
                "final_check": False,
                'parent_order_check': False,
                "med_start": None,
                "med_stop": None,
                'dilution_pair': True
        }

    # Extract rate (preference: fluid parent order -> med params -> defaults)
    rate = None
    rate_source = None
    parent_order_check = False
    final_check = True
    
    # Check fluid parent order first
    if fluid_params["parent_order_check"] and pd.notna(fluid_params.get("rate")) and fluid_params["rate"] != 0:
        rate = fluid_params["rate"]
        parent_order_check = True
        rate_source = "fluid_parent_order"
        print(f"Using fluid parent order rate: {rate}mL/h")
    
    # Check med params
    if med_params["final_check"] and pd.notna(med_params.get("rate")):
        med_rate = med_params["rate"]

        if pd.notna(med_params.get("duration")):
            duration = med_params["duration"]
            volume_rate_from_med = volume / duration

            # If we had a fluid rate, check for exact match
            if rate_source == "fluid_parent_order":
                if volume_rate_from_med == rate:
                    rate_source = "med_params"
                    print(f"Med rate matches fluid rate: {rate}mL/h")
                else:
                    final_check = False
                    print(f"Volume rate from med {volume_rate_from_med} ≠ fluid rate {rate}")
            else:
                rate_source = "med_params"
                rate = volume_rate_from_med
                print(f"Using med rate: {rate}mL/h")
        else:
            print("Med has rate but no duration - cant reconcile it with fluid- CHECK!")
            rate = med_rate
            
        
    # Extract duration based on rate source (needed for fallback calculation)
    duration = None
    if rate_source == "fluid_parent_order":
        duration = fluid_params.get("duration") if pd.notna(fluid_params.get("duration")) else None 
    elif rate_source == "med_params": 
        duration = med_params.get("duration") if pd.notna(med_params.get("duration")) else None
    
    # Fallback: if no rate found but have volume, try to calculate from duration
    if rate is None:
        if pd.notna(volume):
            print(f"No param rate found but have volume: {volume}mL")
            
            # Try to get duration if we don't have it yet
            if duration is None or pd.isna(duration) or duration == 0.0:
                # Try fluid duration
                if pd.notna(fluid_params.get("duration")):
                    duration = fluid_params["duration"]
                    print(f"Using fluid duration: {duration}h")
                # Try med duration
                elif pd.notna(med_params.get("duration")):
                    duration = med_params["duration"]
                    print(f"Using med duration: {duration}h")
            
            # If we have duration now, calculate rate
            if pd.notna(duration) and duration > 0:
                rate = volume / duration
                rate_source = "calculated_from_volume_duration"
                final_check = True
                print(f"Calculated rate from volume and duration: {rate:.2f}mL/h")
            else:
                # No duration - check if small fluid for 1-hour assumption
                # Note: fluid_row is always a fluid in dilution pairs
                if volume <= 1000:
                    duration = 1.0
                    rate = volume
                    rate_source = "small_fluid_1hr_assumption"
                    final_check = True
                    print(f"Small fluid ({volume}mL) - applying 1-hour assumption: rate={rate:.2f}mL/h")
                else:
                    final_check = False
                    print(f"Cannot calculate rate: no duration, large volume ({volume}mL)")
        else:
            final_check = False
            print(f"No param rate or volume found for dilution pair at {action_time}")
    
    # Set default duration if still None
    if duration is None or pd.isna(duration):
        duration = 1.0
        print(f"No duration available, defaulting to 1 hour")
    
    med_start = med_params.get("med_start")
    med_stop = med_params.get("med_stop")
    if med_stop is None:
        med_stop_duration = med_start + pd.TimeDelta(hours = duration)
    
    if verbose:
        print(f"\n[FINAL RESULT]")
        print(f"  Volume: {volume}")
        print(f"  Rate: {rate}")
        print(f"  Duration: {duration}")
        print(f"  Med Start: {med_start}")
        print(f"  Med Stop: {med_stop}")
        print("="*80 + "\n")
    
    return {
        "volume": volume,
        "rate": rate,
        "duration": duration,
        "final_check": final_check,
        'parent_order_check': parent_order_check,
        "med_start": med_start,
        "med_stop": med_stop,
        'dilution_pair': True
    }


def get_order_params_for_row(row, supertable, verbose=False):
    """
    Extract and reconcile medication parameters (volume, rate, duration) from multiple data sources.
    
    This function implements a cascading logic to determine medication parameters with the following priority:
    1. Reconciled parameters from parent order and clinical description
    2. Calculated duration from infusion timestamps (if available)
    3. Derived parameters from infusion rate/amount with unit conversion
    4. Volume-based calculations with 1-hour assumption for small fluids (≤1000mL)
    
    Args:
        row (pd.Series): Medication action row containing:
            - med_action_time: Timestamp of medication action
            - ORDER_PARENT_ORDER: Parent order text
            - ORDER_CLINICAL_DESC: Clinical description text
            - Infusion-related columns (rate_inf, amount_inf, volume_inf, etc.)
        supertable (pd.DataFrame): Patient data table containing daily_weight_kg indexed by time
        verbose (bool): If True, prints detailed logic flow at each decision point
    
    Returns:
        dict: Dictionary containing:
            - volume (float or None): Medication volume in mL
            - rate (float or None): Infusion rate in mL/h
            - duration (float or None): Duration in hours
            - final_check (bool): True if parameters passed validation
            - parent_order_check (bool): True if parent order parameters were valid
            - med_start (pd.Timestamp): Medication start time
            - med_stop (pd.Timestamp or np.nan): Medication stop time
    
    Notes:
        - For small fluids (≤1000mL) without duration info, assumes 1-hour duration
        - NaN values in infusion_params are converted to None for consistency
        - Duration from actual infusion timestamps takes precedence over text-extracted duration
        - Supports unit conversion between different dose/rate units (mg/mcg, etc.)
    """
    if verbose:
        print("\n" + "="*80)
        print(f"PROCESSING ROW: {row.get('med_name', 'Unknown')} at {row.get('med_action_time', 'Unknown')}")
        print(f"Action: {row.get('med_action', 'Unknown')}")
        print("="*80)
    
    # Get patient weight at the time of medication action
    nearest_idx = supertable.index.get_indexer([pd.to_datetime(row["med_action_time"])], method='nearest')[0]
    weight = supertable['daily_weight_kg'].values[nearest_idx]
    
    if verbose:
        print(f"\n[WEIGHT] Patient weight: {weight:.2f} kg")
    
    # Extract parameters from parent orders (clinical desc and manually entered params)
    parent_order_params = extract_parent_order_params(row)
    clinical_desc_params = extract_clinical_desc_params(row)
    
    if verbose:
        print(f"\n[EXTRACTION] Parent order params: {parent_order_params}")
        print(f"[EXTRACTION] Clinical desc params: {clinical_desc_params}")
    
    # Reconcile and validate parameters from both sources
    reconciled_params = reconcile_parameters(parent_order_params, clinical_desc_params)
    
    if verbose:
        print(f"\n[RECONCILIATION] Result: {reconciled_params}")
    
    # Extract parameters from infusion columns (rate_inf, amount_inf, volume_inf, etc.)
    infusion_params = extract_infusion_params(row, weight)
    med_start = pd.to_datetime(row["med_action_time"])

    if verbose:
        print(f"\n[INFUSION PARAMS] Extracted: {infusion_params}")

    # Initialize with reconciled parameters from parent order
    volume = reconciled_params["volume"]
    rate = reconciled_params["rate"]
    duration = reconciled_params["duration"]
    final_check = reconciled_params["final_check"]
    parent_order_check = final_check

    if verbose:
        print(f"\n[INITIALIZATION] Volume: {volume}, Rate: {rate}, Duration: {duration}, Final check: {final_check}")

    # Convert any NaN values in infusion_params to None for consistent handling
    for key in infusion_params.keys():
        if isinstance(infusion_params[key], float) and np.isnan(infusion_params[key]):
            infusion_params[key] = None
    
    # ====================================================================
    # STEP 1: Override duration with actual infusion duration from med_stop if available
    # ====================================================================
    if verbose:
        print(f"\n[STEP 1] Checking for infusion time: med stop exists AND med_stop-med_start is not zero ...")
        print(f"  Condition: med_stop exists = {pd.notna(infusion_params['med_stop'])}")
        print(f"  Condition: duration_inf != 0.0 = {infusion_params['duration_inf'] != 0.0}")
    
    if (pd.notna(infusion_params["med_stop"])) and \
          (infusion_params["duration_inf"] != 0.0):
        # Calculate actual duration from start/stop timestamps
        duration_inf = (infusion_params["med_stop"] - infusion_params["med_start"]).total_seconds() / 3600.0
        
        if verbose:
            print(f"  ✓ CONDITION MET: Have infusion timestamps")
            print(f"  Calculated duration_inf: {duration_inf:.2f} hours")
            print(f"  Reconciled duration: {duration}")
        
        #basically always use infusion duration. These conditional statements wereonly so I could print and check 
        #how often these conditions are met
        if pd.notna(duration) and duration != 0.0:
            # We have both text-extracted and timestamp-based duration
            if verbose:
                print(f"  ✓ Have both durations (text and timestamp)")
            
            if duration_inf != duration:
                duration = duration_inf  # Prefer actual observed duration
                print("Duration mismatch - using infusion duration")
                if verbose:
                    print(f"  → ACTION: Duration mismatch, using infusion duration: {duration:.2f}h")
            else:
                print("Duration match - using reconciled duration")
                if verbose:
                    print(f"  → ACTION: Durations match, keeping: {duration:.2f}h")
        else:
            # No text-extracted duration, use timestamp-based
            print("Using infusion duration")
            if verbose:
                print(f"  ✓ No text-extracted duration")
                print(f"  → ACTION: Using infusion duration: {duration_inf:.2f}h")
            duration = duration_inf
    elif verbose:
        print(f"  ✗ CONDITION NOT MET: No infusion timestamps available")

    # =========================================================================
    # STEP 2: If reconciliation failed (not final_check) or rate is missing, try infusion params
    # =========================================================================
    if verbose:
        print(f"\n[STEP 2] Checking if reconciliation of parent orders failed OR reconciled but rate is missing...")
        print(f"  Condition: final_check = {final_check}")
        print(f"  Condition: rate is missing = {pd.isna(rate)}")
    
    if (not final_check) or (pd.isna(rate)):
        print("Getting infusion params")
        if verbose:
            print(f"  ✓ CONDITION MET: Need to get infusion params (final_check={final_check}, rate={rate})")
        
        # Case 2a: Try to calculate from rate_inf and amount_inf with unit conversion
        if verbose:
            print(f"\n[STEP 2a] Checking rate_unit (from med action dose unit) and amount_inf (from formulary name) ...")
            print(f"  Condition: rate_unit exists = {infusion_params['rate_unit'] is not None}")
            print(f"  Condition: amount_inf exists = {infusion_params['amount_inf'] is not None}")
        
        if (infusion_params["rate_unit"] is not None) and (infusion_params["amount_inf"] is not None):
            if verbose:
                print(f"  ✓ CONDITION MET: Have rate_unit and amount_inf")
            
            units_compatible = False
            converted_amount = infusion_params["amount_inf"]
            
            # Check if units match directly (e.g., "Milligrams" in "Milligrams/Hour")
            if infusion_params["amount_inf_unit_mapped"].lower() in infusion_params["rate_unit"].lower():
                units_compatible = True
                print(f"Direct unit match: {infusion_params['amount_inf_unit_mapped']} in {infusion_params['rate_unit']}")
                if verbose:
                    print(f"  ✓ Direct unit match found")
            else:
                # Try unit conversion (e.g., milligrams to micrograms)
                if verbose:
                    print(f"  No direct unit match, attempting conversion...")
                
                rate_base_unit = infusion_params["rate_unit"].split('/')[0] if '/' in infusion_params["rate_unit"] else infusion_params["rate_unit"]
                
                converted_amount = mpu.convert_units(
                    infusion_params["amount_inf"],
                    infusion_params["amount_inf_unit_mapped"],
                    rate_base_unit
                )
                
                if converted_amount is not None:
                    units_compatible = True
                    print(f"Unit conversion successful: {infusion_params['amount_inf']} {infusion_params['amount_inf_unit_mapped']} -> {converted_amount:.3f} {rate_base_unit}")
                    if verbose:
                        print(f"  ✓ Unit conversion successful")
                else:
                    print(f"Cannot convert {infusion_params['amount_inf_unit_mapped']} to {rate_base_unit}")
                    if verbose:
                        print(f"  ✗ Unit conversion failed")
            
            if units_compatible:
                if verbose:
                    print(f"  ✓ Units compatible, proceeding with calculations")
                
                # Special case: medication stopped (rate = 0)
                if infusion_params["rate_inf"] == 0.0:
                    if verbose:
                        print(f"  ✓ SPECIAL CASE: rate_inf = 0.0 (medication stopped)")
                        print(f"  → ACTION: Setting rate=0, duration=0, volume=0")
                    rate = 0.0
                    duration = 0.0
                    volume = 0.0
                else:
                    # Calculate duration from amount and rate (e.g., 500mg at 100mg/h = 5 hours)
                    duration = converted_amount / infusion_params["rate_inf"]
                    
                    if verbose:
                        print(f"  Calculated duration: {converted_amount:.2f} / {infusion_params['rate_inf']:.2f} = {duration:.2f}h")
                    
                    if not pd.isna(infusion_params["volume_inf"]):
                        # We have volume, calculate rate: rate = volume / duration
                        rate = infusion_params["volume_inf"] / duration
                        final_check = True
                        if verbose:
                            print(f"  ✓ Have volume_inf: {infusion_params['volume_inf']:.2f}mL")
                            print(f"  → ACTION: Calculated rate = {rate:.2f}mL/h, final_check=True")
                    else:
                        final_check = False
                        if verbose:
                            print(f"  ✗ No volume_inf available, final_check=False") 

                # Adjust if bag didn't complete as expected
                if (pd.notna(infusion_params["med_stop"])) & (duration != infusion_params["duration_inf"]):
                    if verbose:
                        print(f"  ✓ Bag incomplete: adjusting to actual duration_inf")
                        print(f"  → ACTION: duration = {infusion_params['duration_inf']:.2f}h, recalculating volume")
                    duration = infusion_params["duration_inf"]
                    volume = rate * duration  
            else:
                final_check = False
                if verbose:
                    print(f"  ✗ Units not compatible, final_check=False")
                
        # Case 2b: We have volume but no rate - try to derive from duration
        elif infusion_params["volume_inf"] is not None:
            if verbose:
                print(f"\n[STEP 2b] Step 2a failed (no rate and amount) but have volume_inf")
                print(f"  Volume_inf: {infusion_params['volume_inf']:.2f}mL")
            
            volume = infusion_params["volume_inf"]
            
            if verbose:
                print(f"  Checking duration: {duration}")
                print(f"  Condition: duration missing or zero = {duration is None or pd.isna(duration) or duration == 0.0}")
            
            if duration is None or pd.isna(duration) or duration == 0.0:
                # No duration from reconciled params, try infusion duration
                if verbose:
                    print(f"  ✓ CONDITION MET: Duration missing/zero")
                    print(f"  Trying infusion duration: {infusion_params['duration_inf']}")
                
                duration = infusion_params["duration_inf"]
                
                if duration is None or pd.isna(duration) or duration == 0.0:
                    # Still no duration - apply 1-hour assumption for small fluids
                    if verbose:
                        print(f"  Still no duration from infusion params")
                        print(f"  Checking 1-hour assumption conditions:")
                        print(f"    - volume <= 1000: {volume <= 1000}")
                        print(f"    - is_fluid: {infusion_params['is_fluid']}")
                    
                    if (volume <= 1000) and (infusion_params["is_fluid"]):
                        duration = 1  # ASSUMPTION: 1 hour default for small fluids
                        rate = volume  # e.g., 500mL over 1hr = 500mL/h
                        final_check = True
                        if verbose:
                            print(f"  ✓ CONDITION MET: Small fluid, applying 1-hour assumption (can be overriden by handle_begin_bag)")
                            print(f"  → ACTION: duration=1h, rate={rate:.2f}mL/h, final_check=True")
                    else:
                        # Large volume or not a fluid - can't make assumptions
                        final_check = False
                        if verbose:
                            print(f"  ✗ CONDITION NOT MET: Large volume or not a fluid")
                            print(f"  → ACTION: Cannot make assumptions, final_check=False")
                else:
                    # We got duration from infusion params, calculate rate
                    rate = volume / duration
                    final_check = True
                    if verbose:
                        print(f"  ✓ Got duration from infusion params: {duration:.2f}h")
                        print(f"  → ACTION: Calculated rate = {rate:.2f}mL/h, final_check=True")
            else:
                # We have both volume and duration from reconciled params
                rate = volume / duration
                final_check = True
                if verbose:
                    print(f"  ✓ Have both volume and duration from reconciled params")
                    print(f"  → ACTION: Calculated rate = {rate:.2f}mL/h, final_check=True")
        else:
            # No usable infusion parameters
            final_check = False
            if verbose:
                print(f"\n[STEP 2 - NO MATCH] No usable infusion parameters")
                print(f"  → ACTION: final_check=False")
    elif verbose:
        print(f"  ✗ CONDITION NOT MET: Already have valid params (final_check={final_check}, rate={rate})") 
    
    # ====================================================================
    # STEP 3: Determine medication stop time
    # ====================================================================
    if verbose:
        print(f"\n[STEP 3] Determining medication stop time...")
        print(f"  Checking conditions for med_stop:")
        print(f"    - infusion med_stop exists: {pd.notna(infusion_params['med_stop'])}")
        print(f"    - duration exists and != 0: {pd.notna(duration) and duration != 0.0}")
    
    if (pd.notna(infusion_params["med_stop"])):
        # Use actual stop time from infusion data
        med_stop = infusion_params["med_stop"]
        if verbose:
            print(f"  ✓ CONDITION MET: Using actual stop time from infusion data")
            print(f"  → ACTION: med_stop = {med_stop}")
    elif pd.notna(duration) and duration != 0.0:
        # Calculate stop time from start + duration
        med_stop = med_start + pd.Timedelta(hours=duration)
        if verbose:
            print(f"  ✓ CONDITION MET: Have duration, calculating stop time")
            print(f"  → ACTION: med_stop = med_start + {duration:.2f}h = {med_stop}")
    else:
        # No duration available, stop time unknown
        med_stop = np.nan
        if verbose:
            print(f"  ✗ NO CONDITION MET: No stop time available")
            print(f"  → ACTION: med_stop = NaN")
    
    print(f"Med start: {med_start}, Med stop: {med_stop}, Volume: {volume}, Rate: {rate}, Duration: {duration}, Final Check: {final_check}")
    
    if verbose:
        print(f"\n[FINAL RESULT]")
        print(f"  Volume: {volume}")
        print(f"  Rate: {rate}")
        print(f"  Duration: {duration}")
        print(f"  Final Check: {final_check}")
        print(f"  Parent Order Check: {parent_order_check}")
        print(f"  Med Start: {med_start}")
        print(f"  Med Stop: {med_stop}")
        print("="*80 + "\n")
    
    return {
        "volume": volume,
        "rate": rate,
        "duration": duration,
        "final_check": final_check,
        'parent_order_check': parent_order_check,
        "med_start": med_start,
        "med_stop": med_stop,
        "dilution_pair": False
    }


################### PARAMETER EXTRACTION ################

def extract_infusion_params(row, weight):
    """
    Extract infusion parameters from a single medication row just from the infusion meds columns
    
    Args:
        row: Single row from medications DataFrame
        weight: Patient weight
        
    Returns:
        dict: Processed medication data with extracted parameters
    """
    # Process start/stop times if available
    med_start = None
    med_stop = None
    duration = None #this is the duration from the infusion meds columns
    volume = None #this is the volume from the infusion meds columns
    volume_unit = None #this is the unit of the volume
    rate = None #this is the rate from the infusion meds columns
    rate_unit = None #this is the unit of the rate
    amount = None #this is the amount from the infusion meds columns
    amount_unit = None #this is the unit of the amount
    amount_unit_mapped = None 
    is_fluid = row["is_fluid"]

    

    if not pd.isna(row["med_start"]):
        med_start = pd.to_datetime(row["med_start"])
    
        if not pd.isna(row["med_stop"]):
            med_stop = pd.to_datetime(row["med_stop"])
            duration = (med_stop - med_start).total_seconds() / 3600.0
    
    if row["is_infusion"] or (row["med_action"] in ["Infuse", "Begin Bag", "Rate Change"]):
        if row["volume_inf"] is not None:
            volume = row["volume_inf"]*1000 if row["volume_inf_unit"] == "L" else row["volume_inf"]
            volume_unit = row["volume_inf_unit"]
        if row["amount_inf"] is not None:
            amount = row["amount_inf"]
            amount_unit = row["amount_inf_unit"] 
            amount_unit_mapped = mpu.amount_unit_mapping[amount_unit] if amount_unit in mpu.amount_unit_mapping else amount_unit

        if row["med_action_dose"] is not None:
            results = mpu.calculate_dose_based_rate(row, weight)
            if results["rate"] is not None:
                rate = results["rate"]
                rate_unit = results["rate_unit"]
        return {
            'med_start': med_start,
            'med_stop': med_stop,
            'volume_inf': volume,
            'rate_inf': rate,
            'duration_inf': duration,
            'amount_inf': amount,
            'amount_inf_unit': amount_unit,
            'amount_inf_unit_mapped': amount_unit_mapped,
            'volume_unit': volume_unit,
            'rate_unit': rate_unit,
            'is_fluid': is_fluid
        }
    else:
        return {
            'med_start': None,
            'med_stop': None,
            'volume_inf': None,
            'rate_inf': None,
            'duration_inf': None,
            'amount_inf': None,
            'amount_inf_unit': None,
            'volume_unit': None,
            'rate_unit': None,
            'is_fluid': None
        } 

def check_duration_in_desc(row):
    """
    Check if duration from parent order (from ORDERED_MEDS) matches description text (from OUT3).
    
    Args:
        row: DataFrame row containing DURATION_INFUSION_ORDERED_QTY and ORDER_CLINICAL_DESC
        
    Returns:
        dict: {"found": bool, "unit": str|None}
    """
    duration = row["DURATION_INFUSION_ORDERED_QTY"]
    desc = row["ORDER_CLINICAL_DESC"]
    
    if pd.isna(duration) or pd.isna(desc):
        return {"found": False, "unit": None}
    
    desc = desc.lower()
    
    # Check for hours - handle both integer and decimal
    if duration == int(duration):
        hour_pattern = f"{int(duration)} hr"
    else:
        hour_pattern = f"{duration} hr"
    
    if hour_pattern in desc:
        return {"found": True, "unit": "hours"}
    
    # Check for minutes - handle both integer and decimal
    if duration == int(duration):
        minute_pattern = f"{int(duration)} minute"
    else:
        minute_pattern = f"{duration} minute"
    
    if minute_pattern in desc:
        return {"found": True, "unit": "minutes"}
    
    return {"found": False, "unit": None}


def extract_parent_order_params(row):
    """
    Extract infusion parameters from parent order data (ORDERED_MEDS).
    
    Args:
        row: DataFrame row with parent order information
        
    Returns:
        dict: Contains volume (in ml), rate (in ml/hr), duration (in hrs), and suspicious flag (in case of weird units in the raw data)
    """
    if not row["ORDER_PARENT_ID"] or pd.isna(row["ORDER_PARENT_ID"]):
        return {"has_parent": False, "suspicious": False}
    
    duration_parent = row["DURATION_INFUSION_ORDERED_QTY"]
    volume_parent = row["VOLUME_ORDERED_QTY"]
    volume_unit = row["VOLUME_UNIT_MEASURE"]
    rate_parent = row["INFUSION_ORDER_RT"]
    rate_unit = row["RATE_UNIT_MEASURE"]
    
    suspicious = False
    
    # Check for unexpected units
    if volume_unit != "Milliliter":
        suspicious = True
        volume_parent = np.nan 
    if rate_unit != "Milliliter/hour":
        suspicious = True
        rate_parent = np.nan 
    
    # Check duration consistency
    duration_check = check_duration_in_desc(row)
    if duration_check["found"]:
        if duration_check["unit"] == "minutes":
            duration_parent = duration_parent / 60.0
    # I'm always going to believe the volume from the parent order 
    #elif (pd.isna(rate_parent) and pd.isna(volume_parent) and 
    #      abs(duration_parent - (volume_parent / rate_parent)) > 10):
    #    suspicious = True 
    
    if pd.isna(rate_parent) or (rate_parent == 0.0):
        if (pd.notna(volume_parent) and pd.notna(duration_parent) and (duration_parent != 0.0)):
            rate_parent = volume_parent/duration_parent

    
    return {
        "has_parent": True,
        "volume": volume_parent,
        "rate": rate_parent,
        "duration": duration_parent,
        "suspicious": suspicious
    }


def extract_clinical_desc_params(row):
    """
    Extract infusion parameters from clinical description text (from OUT3).
    
    Args:
        row: DataFrame row with ORDER_CLINICAL_DESC
        
    Returns:
        dict: Contains volume, rate, duration, and sus flag
    """
    if not row["ORDER_CLINICAL_DESC"] or pd.isna(row["ORDER_CLINICAL_DESC"]):
        return {"has_clinical_desc": False, "sus": False}
    
    params = mpu.parse_clinical_description(row["ORDER_CLINICAL_DESC"])
    sus = False
    
    # Helper function to check if value is valid (not None and not NaN and not nothing)
    def is_valid_param(param):
        if param is None:
            return False
        if isinstance(param, list):
            return len(param) > 0
        return not pd.isna(param)
    
    # Handle multiple volumes
    # Sometimes the desc says "total volume". That is parsed into "volume_unit" by the parser code.
    # So that is chosen as the volume if there are multiple volumes. Else, the first volume is chosen.
    # This is brittle, but not worth more time 
    if params["volume"]:
        if len(params["volume"]) > 1:
            if 'total_volume' in params['volume_unit']:
                vol_idx = params["volume_unit"].index('total_volume')
                params['volume'] = params["volume"][vol_idx]
                params['volume_unit'] = 'total_volume'
            else:
                sus = True
        else:
            params["volume"] = params["volume"][0]
            params["volume_unit"] = params["volume_unit"][0]
    else:
        params["volume"] = None
    
    # Handle multiple rates - sus is true 
    if params["rate"]:
        if len(params["rate"]) > 1:
            sus = True
        else:
            params["rate"] = params["rate"][0]
    else:
        params["rate"] = None
    
    # Handle multiple durations - use rate and volume to check the right duration
    if params["duration"]:
        params["duration"] = list(set(params["duration"]))
        if len(params["duration"]) > 1 and 0.0 in params["duration"]:
            params["duration"].remove(0.0)
        
        if len(params["duration"]) > 1:
            if (not sus and is_valid_param(params["volume"]) and 
                is_valid_param(params["rate"]) and not isinstance(params["volume"], list) and 
                not isinstance(params["rate"], list)):
                calculated_duration = params["volume"] / params["rate"]
                if calculated_duration in params["duration"]:
                    params["duration"] = calculated_duration
                else:
                    sus = True
        else:
            params["duration"] = params["duration"][0]
    else:
        params["duration"] = None
    
    # Use other two to check rate 
    if not is_valid_param(params["rate"]) or (params["rate"] == 0.0):
        if (is_valid_param(params["volume"]) and is_valid_param(params["duration"])):
            if params["duration"] == 0.0:
                params["rate"] = params["volume"]
            else:
                params["rate"] = params["volume"] / params["duration"]
        elif ("now" in row["ORDER_CLINICAL_DESC"].lower()) and is_valid_param(params["volume"]):
            params["rate"] = params["volume"]
    
    # Use other two to check volume
    if not is_valid_param(params["volume"]) or (isinstance(params["volume"], list) and len(params["volume"]) > 1):
        if (is_valid_param(params["duration"]) and is_valid_param(params["rate"])):
            params["volume"] = params["duration"] * params["rate"] 


    # Final consistency check
    if (not sus and is_valid_param(params["volume"]) and is_valid_param(params["rate"]) and 
        is_valid_param(params["duration"]) and
        abs(params["duration"] - (params["volume"] / params["rate"])) > 10):
        sus = True
    
    return {
        "has_clinical_desc": True,
        "volume": params["volume"],
        "rate": params["rate"],
        "duration": params["duration"],
        "sus": sus
    }

###### VALIDATE AND RECONCILE MULTIPLE PARAMETER SOURCES ######################

def validate_parameter_consistency(volume, rate, duration, tolerance=10):
    """
    Check if volume, rate, and duration are consistent with each other.
    
    Args:
        volume: Volume value
        rate: Rate value  
        duration: Duration value
        tolerance: Acceptable difference in hours for duration calculation
        
    Returns:
        bool: True if parameters are consistent or if insufficient data to check
    """
    # Check if all parameters are valid (not None and not NaN)
    if (not volume or pd.isna(volume) or 
        not rate or pd.isna(rate) or 
        not duration or pd.isna(duration)):
        return True  # Can't validate incomplete sets
    
    try:
        calculated_duration = volume / rate
        return abs(duration - calculated_duration) <= tolerance
    except (ZeroDivisionError, TypeError):
        return False


def validate_param_against_list(param_value, param_list, tolerance=0.1):
    """
    Check if a parameter value exists in a list of parameters within tolerance.
    
    Args:
        param_value: Single parameter value to check
        param_list: List of parameter values to check against
        tolerance: Relative tolerance for matching (default 10%)
        
    Returns:
        bool: True if param_value matches any value in param_list within tolerance
    """
    if (not param_value or pd.isna(param_value) or not param_list):
        return False
        
    if not isinstance(param_list, list):
        param_list = [param_list]
        
    for list_val in param_list:
        if list_val and not pd.isna(list_val) and abs(param_value - list_val) <= (param_value * tolerance):
            return True
    return False


def cross_validate_parameters(parent_data, clinical_data, tolerance=10):
    """
    Cross-validate parameters between parent and clinical data sources.
    Enhanced to handle cases where clinical params might be lists.
    
    Args:
        parent_data: Dictionary from extract_parent_order_params
        clinical_data: Dictionary from extract_clinical_desc_params
        tolerance: Tolerance for duration consistency check
        
    Returns:
        dict: Contains validation results and recommendations
    """
    p_vol = parent_data.get("volume")
    p_rate = parent_data.get("rate") 
    p_dur = parent_data.get("duration")
    
    # Get clinical parameters (might be lists if suspicious)
    c_vol = clinical_data.get("volume")
    c_rate = clinical_data.get("rate")
    c_dur = clinical_data.get("duration")
    
    # Check internal consistency for parent (always single values)
    parent_consistent = validate_parameter_consistency(p_vol, p_rate, p_dur, tolerance)
    
    # For clinical, check consistency if parameters are single values
    clinical_consistent = True
    if (isinstance(c_vol, (int, float)) and isinstance(c_rate, (int, float)) and 
        isinstance(c_dur, (int, float))):
        clinical_consistent = validate_parameter_consistency(c_vol, c_rate, c_dur, tolerance)
    
    # Enhanced cross-consistency check that handles lists
    cross_consistent = True
    parent_validates_clinical = True
    
    # Check if parent volume exists in clinical volume list
    if p_vol and c_vol:
        if isinstance(c_vol, list):
            if not validate_param_against_list(p_vol, c_vol):
                parent_validates_clinical = False
        else:
            if abs(p_vol - c_vol) > (p_vol * 0.1):
                cross_consistent = False
    
    # Check if parent rate exists in clinical rate list
    if p_rate and c_rate:
        if isinstance(c_rate, list):
            if not validate_param_against_list(p_rate, c_rate):
                parent_validates_clinical = False
        else:
            if abs(p_rate - c_rate) > (p_rate * 0.1):
                cross_consistent = False
    
    # Check if parent duration exists in clinical duration list
    if p_dur and c_dur:
        if isinstance(c_dur, list):
            if not validate_param_against_list(p_dur, c_dur, tolerance/p_dur if p_dur > 0 else 0.1):
                parent_validates_clinical = False
        else:
            if abs(p_dur - c_dur) > tolerance:
                cross_consistent = False
    
    # Calculate completeness (excluding NaN values)
    parent_completeness = sum([
        bool(p_vol and not pd.isna(p_vol)), 
        bool(p_rate and not pd.isna(p_rate)), 
        bool(p_dur and not pd.isna(p_dur))
    ])
    
    # For clinical completeness, count non-empty, non-NaN parameters
    clinical_completeness = 0
    if c_vol and not pd.isna(c_vol) and (not isinstance(c_vol, list) or len(c_vol) > 0):
        clinical_completeness += 1
    if c_rate and not pd.isna(c_rate) and (not isinstance(c_rate, list) or len(c_rate) > 0):
        clinical_completeness += 1  
    if c_dur and not pd.isna(c_dur) and (not isinstance(c_dur, list) or len(c_dur) > 0):
        clinical_completeness += 1
    
    return {
        "parent_consistent": parent_consistent,
        "clinical_consistent": clinical_consistent,
        "cross_consistent": cross_consistent,
        "parent_validates_clinical": parent_validates_clinical,
        "parent_completeness": parent_completeness,
        "clinical_completeness": clinical_completeness
    }


def fill_missing_parameters(primary_data, secondary_data):
    """
    Fill missing parameters from secondary source.
    
    Args:
        primary_data: Primary parameter set (dict)
        secondary_data: Secondary parameter set to fill gaps from (dict)
        
    Returns:
        dict: Combined parameter set
    """
    result = {
        "volume": primary_data.get("volume") or secondary_data.get("volume"),
        "rate": primary_data.get("rate") or secondary_data.get("rate"),
        "duration": primary_data.get("duration") or secondary_data.get("duration")
    }
    
    # If we have volume and rate but no duration, calculate it
    if result["volume"] and result["rate"] and not result["duration"]:
        try:
            result["duration"] = result["volume"] / result["rate"]
        except ZeroDivisionError:
            pass
    
    return result


def reconcile_parameters(parent_data, clinical_data):
    """
    Reconcile parameters from parent order and clinical description using cross-validation.
    
    Args:
        parent_data: Dictionary from extract_parent_order_params
        clinical_data: Dictionary from extract_clinical_desc_params
        
    Returns:
        dict: Final reconciled parameters
    """
    has_parent = parent_data.get("has_parent", False)
    parent_sus = parent_data.get("suspicious", False)
    has_clinical = clinical_data.get("has_clinical_desc", False)
    clinical_sus = clinical_data.get("sus", False)
    
    # Early exit cases - only one source available
    if not has_parent and not has_clinical:
        return {
            "volume": None, "rate": None, "duration": None,
            "source": "no_data", "final_check": False
        }
    
    if not has_parent and has_clinical:
        if not clinical_sus:
            return {
                "volume": clinical_data["volume"],
                "rate": clinical_data["rate"], 
                "duration": clinical_data["duration"],
                "source": "clinical_only", "final_check": True
            }
        else:
            return {
                "volume": clinical_data["volume"],
                "rate": clinical_data["rate"],
                "duration": clinical_data["duration"], 
                "source": "clinical_only_suspicious", "final_check": False
            }
    
    if has_parent and not has_clinical:
        if not parent_sus:
            return {
                "volume": parent_data["volume"],
                "rate": parent_data["rate"],
                "duration": parent_data["duration"],
                "source": "parent_only", "final_check": True
            }
        else:
            return {
                "volume": parent_data["volume"],
                "rate": parent_data["rate"], 
                "duration": parent_data["duration"],
                "source": "parent_only_suspicious", "final_check": False
            }
    
    # Both sources available - perform cross-validation
    validation = cross_validate_parameters(parent_data, clinical_data)
    
    # Case 1: Both clean - prefer parent, but use cross-validation
    if not parent_sus and not clinical_sus:
        if validation["cross_consistent"]:
            # Fill any missing parent params from clinical
            final_params = fill_missing_parameters(parent_data, clinical_data)
            return {
                "volume": final_params["volume"],
                "rate": final_params["rate"],
                "duration": final_params["duration"],
                "source": "parent_clinical_consistent", "final_check": True
            }
        else:
            # Prefer the more complete set
            if validation["parent_completeness"] >= validation["clinical_completeness"]:
                return {
                    "volume": parent_data["volume"],
                    "rate": parent_data["rate"],
                    "duration": parent_data["duration"],
                    "source": "parent_over_clinical", "final_check": True
                }
            else:
                return {
                    "volume": clinical_data["volume"],
                    "rate": clinical_data["rate"],
                    "duration": clinical_data["duration"],
                    "source": "clinical_over_parent", "final_check": True
                }
    
    # Case 2: Parent clean, clinical suspicious - validate clinical against parent
    elif not parent_sus and clinical_sus:
        if validation["parent_validates_clinical"]:
            # Parent parameters exist in clinical lists - clinical is likely correct
            final_params = fill_missing_parameters(parent_data, clinical_data)
            return {
                "volume": final_params["volume"],
                "rate": final_params["rate"], 
                "duration": final_params["duration"],
                "source": "parent_validated_clinical_lists", "final_check": True
            }
        elif validation["cross_consistent"]:
            # Clinical validates against parent - use combined
            final_params = fill_missing_parameters(parent_data, clinical_data)
            return {
                "volume": final_params["volume"],
                "rate": final_params["rate"], 
                "duration": final_params["duration"],
                "source": "parent_validated_clinical", "final_check": True
            }
        else:
            # Use parent, fill missing from clinical if needed
            final_params = fill_missing_parameters(parent_data, clinical_data)
            return {
                "volume": final_params["volume"],
                "rate": final_params["rate"],
                "duration": final_params["duration"], 
                "source": "parent_with_clinical_fill", "final_check": True
            }
    
    # Case 3: Parent suspicious, clinical clean - validate parent against clinical  
    elif parent_sus and not clinical_sus:
        if validation["cross_consistent"]:
            # Parent validates against clinical - use combined
            final_params = fill_missing_parameters(clinical_data, parent_data)
            return {
                "volume": final_params["volume"],
                "rate": final_params["rate"],
                "duration": final_params["duration"],
                "source": "clinical_validated_parent", "final_check": True
            }
        else:
            # Use clinical, fill missing from parent if needed
            final_params = fill_missing_parameters(clinical_data, parent_data)
            return {
                "volume": final_params["volume"],
                "rate": final_params["rate"],
                "duration": final_params["duration"],
                "source": "clinical_with_parent_fill", "final_check": True
            }
    
    # Case 4: Both suspicious - mutual validation
    else:  # parent_sus and clinical_sus
        if validation["cross_consistent"]:
            # They agree despite being individually suspicious - likely correct
            final_params = fill_missing_parameters(
                parent_data if validation["parent_completeness"] >= validation["clinical_completeness"] 
                else clinical_data,
                clinical_data if validation["parent_completeness"] >= validation["clinical_completeness"]
                else parent_data
            )
            return {
                "volume": final_params["volume"],
                "rate": final_params["rate"],
                "duration": final_params["duration"],
                "source": "mutually_validated_suspicious", "final_check": True
            }
        else:
            # Choose the more internally consistent or complete set
            if validation["parent_consistent"] and not validation["clinical_consistent"]:
                final_params = fill_missing_parameters(parent_data, clinical_data)
                source = "parent_suspicious_consistent"
            elif validation["clinical_consistent"] and not validation["parent_consistent"]:
                final_params = fill_missing_parameters(clinical_data, parent_data)
                source = "clinical_suspicious_consistent"
            elif validation["parent_completeness"] > validation["clinical_completeness"]:
                final_params = fill_missing_parameters(parent_data, clinical_data)
                source = "parent_suspicious_more_complete"
            elif validation["clinical_completeness"] > validation["parent_completeness"]:
                final_params = fill_missing_parameters(clinical_data, parent_data)
                source = "clinical_suspicious_more_complete"
            else:
                # Last resort - prefer parent
                final_params = fill_missing_parameters(parent_data, clinical_data)
                source = "both_suspicious_prefer_parent"
            
            return {
                "volume": final_params["volume"],
                "rate": final_params["rate"],
                "duration": final_params["duration"],
                "source": source, "final_check": False
            }
