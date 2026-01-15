from pathlib import Path
import pandas as pd
import numpy as np
import sys
sys.path.append("../")
import data_preprocess.medication_processor as mp
import data_preprocess.parameter_extractor as pe
    
import data_preprocess.med_processing_utils as mpu


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

def process_dilution_pair_with_common_timestamps(
    fluid_rows,
    med_rows,
    order_id,
    combined_med_name,
    medsdict,
    supertable
):
    """
    Process a dilution pair with common timestamps.
    
    Args:
        fluid_rows: DataFrame for fluid medication
        med_rows: DataFrame for non-fluid medication
        order_id: Order ID
        combined_med_name: for indexing the medsdict
        medsdict: Medication dictionary
        supertable: Patient data table

    Returns:
        tuple: (updated_medsdict, updated_processed_indices)
    """
    fluid_rows = fluid_rows.sort_values("med_action_time")
    med_rows = med_rows.sort_values("med_action_time")
    
    processed_indices = set()
    orphaned_infuse_rows = []

    print(f"Processing dilution pair {combined_med_name}")
    med_rows['med_action_time'] = pd.to_datetime(med_rows['med_action_time'])
    fluid_rows['med_action_time'] = pd.to_datetime(fluid_rows['med_action_time'])

    for idx, row in med_rows.iterrows():
        if row.name in processed_indices:
            continue
            
        med_action = row["med_action"]
        med_action_time = row["med_action_time"]
        fluid_row = fluid_rows.loc[fluid_rows["med_action_time"] == med_action_time].iloc[0]
        print(f"Processing {med_action} at {med_action_time}")
        
        if med_action == "Begin Bag":
            processed_indices.add(row.name)

            params = pe.get_order_params_for_dilution_pairs(fluid_row, row, supertable)
            print("FINAL PARAMS")
            print(params)
            # Use unified Begin Bag handler
            medsdict, processed_indices = mp.handle_begin_bag(
                row, med_rows, processed_indices, order_id, combined_med_name, medsdict, supertable,
                is_explicit=True, action_label="BEGIN BAG", params = params
            )
                    
        elif med_action == "Rate Change":
            processed_indices.add(row.name)
            print("Processing Rate Change")
            params = pe.get_order_params_for_dilution_pairs(fluid_row, row, supertable)
            # Get current medication state
            if not medsdict[order_id][combined_med_name]["set"]:
                print("First action is Rate Change - treating as implicit Begin Bag")
                medsdict, processed_indices = mp.handle_implicit_begin_bag(
                    row, med_rows, processed_indices, order_id, combined_med_name, medsdict, supertable, params = params
                )
                continue
            
            medsdict, processed_indices = mp.handle_rate_change(
                row, med_rows, processed_indices, order_id, combined_med_name, medsdict, supertable, params = params
            )
                
        elif med_action == "Infuse":
            # Check if this is the first action and no medication is set up yet
            if not medsdict[order_id][combined_med_name]["set"] and row.name not in processed_indices:
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
                    params = pe.get_order_params_for_dilution_pairs(fluid_row, row, supertable)
                    medsdict, processed_indices = mp.handle_implicit_begin_bag(
                        row, med_rows, processed_indices, order_id, combined_med_name, medsdict, supertable, params = params
                    )
                    continue
                
            # Check if this is an orphaned infuse action
            if row.name not in processed_indices:
                # This infuse wasn't consumed by any Begin Bag
                current_time = med_action_time
                
                # Check if there's an active infusion that should cover this time
                active_infusion_found = False
                if medsdict[order_id][combined_med_name]["set"]:
                    for i, (start, stop) in enumerate(zip(
                        medsdict[order_id][combined_med_name]["med_start"],
                        medsdict[order_id][combined_med_name]["med_stop"]
                    )):
                        if pd.notna(start) and pd.notna(stop):
                            if start <= current_time <= stop:
                                active_infusion_found = True
                                break
                
                if not active_infusion_found:
                    # This is an orphaned infuse - treat as implicit begin bag
                    print(f"ORPHANED INFUSE: {combined_med_name} at {current_time} - treating as implicit Begin Bag")
                    params = pe.get_order_params_for_dilution_pairs(fluid_row, row, supertable)
                    medsdict, processed_indices = mp.handle_implicit_begin_bag(
                        row, med_rows, processed_indices, order_id, combined_med_name, medsdict, supertable, params = params
                    )
                    continue
                else:
                    # This infuse is within an active period, mark as processed
                    processed_indices.add(row.name)
        
        elif med_action == "Not Recorded":
            processed_indices.add(row.name)
            print("Processing Not Recorded action")
            params = pe.get_order_params_for_dilution_pairs(fluid_row, row, supertable)
            medsdict, processed_indices = mp.handle_not_recorded(
                row, processed_indices, order_id, combined_med_name, medsdict, supertable, params = params
            )

        elif med_action == "Bolus":
            processed_indices.add(row.name)
            print("Processing Bolus action")
            params = pe.get_order_params_for_dilution_pairs(fluid_row, row, supertable)
            medsdict, processed_indices = mp.handle_bolus(
                row, processed_indices, order_id, combined_med_name, medsdict, supertable, params = params
            )
        else:
            # Handle other unrecorded med actions (not "Begin Bag", "Rate Change", "Infuse", "Not Recorded", or "Bolus")
            if not medsdict[order_id][med_name]["set"] and row.name not in processed_indices:
                print(f"Other unrecorded med action: {med_action} - checking if params are valid")
                params = pe.get_order_params_for_dilution_pairs(fluid_row, row, supertable)
                
                if params["final_check"]:
                    print(f"Valid params found for unrecorded action {med_action} - treating as implicit Begin Bag")
                    medsdict, processed_indices = mp.handle_implicit_begin_bag(
                        row, med_rows, processed_indices, order_id, combined_med_name, medsdict, supertable, params = params
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
    if medsdict[order_id][combined_med_name]["set"]:
        num_periods = len(medsdict[order_id][combined_med_name]["med_start"])
        print(f"\n=== MEDICATION SUMMARY: {combined_med_name} ===")
        print(f"Total infusion periods: {num_periods}")
        for i in range(num_periods):
            start = medsdict[order_id][combined_med_name]["med_start"][i]
            stop = medsdict[order_id][combined_med_name]["med_stop"][i]
            rate = medsdict[order_id][combined_med_name]["rate"][i]
            volume = medsdict[order_id][combined_med_name]["volume"][i]
            duration = medsdict[order_id][combined_med_name]["duration"][i]
            print(f"  Period {i+1}: {start} to {stop} | Rate: {rate:.1f}mL/h | Volume: {volume:.1f}mL | Duration: {duration:.2f}h")
        print("=====================================\n")
    else:
        print(f"\n=== MEDICATION SUMMARY: {combined_med_name} ===")
        print("No valid infusion periods established")
        print("=====================================\n")
    
    return medsdict, processed_indices


def process_dilution_pairs(
    order_id,
    fluid_pair_key,
    med_pair_key,
    order_rows,
    medsdict,
    supertable
):
    """
    Process dilution pairs with separate handling for common and non-common timestamps.
    
    Args:
        order_id: Order ID
        fluid_pair_key: (med_name, formulary_name) for fluid
        med_pair_key: (med_name, formulary_name) for medication
        order_rows: All rows for this order_id
        medsdict: previous meds
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
            
    processed_indices = set()
    
    # Get common and non-common timestamps
    common_times, fluid_only_times, med_only_times = get_common_and_noncommon_timestamps(fluid_rows, med_rows)
    
    print(f"Found {len(common_times)} common timestamps, {len(fluid_only_times)} fluid-only, {len(med_only_times)} med-only")
    
    # Create combined medication name for dilution pairs
    combined_med_name = mpu.create_combined_med_name(med_pair_key[0], fluid_pair_key[0])
    
    # Process common timestamps - filter to only common timestamp rows
    if common_times:
        fluid_common_rows = fluid_rows[pd.to_datetime(fluid_rows['med_action_time']).isin(common_times)]
        med_common_rows = med_rows[pd.to_datetime(med_rows['med_action_time']).isin(common_times)]
        
        print(f"Processing {len(fluid_common_rows)} fluid rows and {len(med_common_rows)} med rows at common timestamps")
        
        
        medsdict[order_id][combined_med_name] = {
                    'rate': [], 'duration': [], 'med_start': [], 'med_stop': [], 'volume': [], 'set': False,
                    'original_rate': None, 'original_volume': None, 'original_duration': None
                }

        medsdict, processed_indices = process_dilution_pair_with_common_timestamps(
            fluid_common_rows, med_common_rows, order_id, combined_med_name,
            medsdict, supertable
        )
    
    # Process non-common timestamps as single infusions
    
    # Process fluid-only timestamps
    if fluid_only_times:
        print(f"\nProcessing {len(fluid_only_times)} fluid-only timestamps")
        
        # Filter to only non-common timestamp rows
        fluid_only_rows = fluid_rows[pd.to_datetime(fluid_rows['med_action_time']).isin(fluid_only_times)]
        
        if len(fluid_only_rows) > 0:
            # Initialize medsdict entry for this fluid
            if fluid_pair_key[0] not in medsdict[order_id]:
                medsdict[order_id][fluid_pair_key[0]] = {
                    'rate': [], 'duration': [], 'med_start': [], 'med_stop': [], 'volume': [], 'set': False,
                    'original_rate': None, 'original_volume': None, 'original_duration': None
                }
            
            # Process as normal medication timeline
            print(f"Processing {len(fluid_only_rows)} fluid-only rows as single medication")
            medsdict, fluid_processed = mp.process_medication_timeline_new(
                order_id, 
                fluid_pair_key[0],  # med_name
                fluid_only_rows,
                supertable, 
                medsdict
            )
            processed_indices.update(fluid_processed)
            print(f"Successfully processed fluid-only timestamps")
    
    # Process med-only timestamps
    if med_only_times:
        print(f"\nProcessing {len(med_only_times)} med-only timestamps")
        
        # Filter to only non-common timestamp rows
        med_only_rows = med_rows[pd.to_datetime(med_rows['med_action_time']).isin(med_only_times)]
        
        if len(med_only_rows) > 0:
            # Initialize medsdict entry for this medication
            if med_pair_key[0] not in medsdict[order_id]:
                medsdict[order_id][med_pair_key[0]] = {
                    'rate': [], 'duration': [], 'med_start': [], 'med_stop': [], 'volume': [], 'set': False,
                    'original_rate': None, 'original_volume': None, 'original_duration': None
                }
            
            # Process as normal medication timeline
            print(f"Processing {len(med_only_rows)} med-only rows as single medication")
            medsdict, med_processed = mp.process_medication_timeline_new(
                order_id, 
                med_pair_key[0],  # med_name
                med_only_rows,
                supertable, 
                medsdict
            )
            processed_indices.update(med_processed)
            print(f"Successfully processed med-only timestamps")
    
    # Summary - print for all processed medications
    print(f"\n{'='*80}")
    print(f"DILUTION PAIR PROCESSING SUMMARY")
    print(f"{'='*80}")
    
    # Summary for combined dilution pair (common timestamps)
    if combined_med_name in medsdict[order_id] and medsdict[order_id][combined_med_name]["set"]:
        num_periods = len(medsdict[order_id][combined_med_name]["med_start"])
        print(f"\n=== COMBINED DILUTION PAIR: {combined_med_name} ===")
        print(f"Total infusion periods: {num_periods}")
        for i in range(num_periods):
            start = medsdict[order_id][combined_med_name]["med_start"][i]
            stop = medsdict[order_id][combined_med_name]["med_stop"][i]
            rate = medsdict[order_id][combined_med_name]["rate"][i]
            volume = medsdict[order_id][combined_med_name]["volume"][i]
            duration = medsdict[order_id][combined_med_name]["duration"][i]
            print(f"  Period {i+1}: {start} to {stop} | Rate: {rate:.1f}mL/h | Volume: {volume:.1f}mL | Duration: {duration:.2f}h")
    else:
        print(f"\n=== COMBINED DILUTION PAIR: {combined_med_name} ===")
        print("No common timestamp infusions")
    
    # Summary for fluid-only timestamps
    if fluid_pair_key[0] in medsdict[order_id] and medsdict[order_id][fluid_pair_key[0]]["set"]:
        num_periods = len(medsdict[order_id][fluid_pair_key[0]]["med_start"])
        print(f"\n=== FLUID ONLY ({fluid_pair_key[0]}) ===")
        print(f"Total infusion periods: {num_periods}")
        for i in range(num_periods):
            start = medsdict[order_id][fluid_pair_key[0]]["med_start"][i]
            stop = medsdict[order_id][fluid_pair_key[0]]["med_stop"][i]
            rate = medsdict[order_id][fluid_pair_key[0]]["rate"][i]
            volume = medsdict[order_id][fluid_pair_key[0]]["volume"][i]
            duration = medsdict[order_id][fluid_pair_key[0]]["duration"][i]
            print(f"  Period {i+1}: {start} to {stop} | Rate: {rate:.1f}mL/h | Volume: {volume:.1f}mL | Duration: {duration:.2f}h")
    
    # Summary for med-only timestamps
    if med_pair_key[0] in medsdict[order_id] and medsdict[order_id][med_pair_key[0]]["set"]:
        num_periods = len(medsdict[order_id][med_pair_key[0]]["med_start"])
        print(f"\n=== MEDICATION ONLY ({med_pair_key[0]}) ===")
        print(f"Total infusion periods: {num_periods}")
        for i in range(num_periods):
            start = medsdict[order_id][med_pair_key[0]]["med_start"][i]
            stop = medsdict[order_id][med_pair_key[0]]["med_stop"][i]
            rate = medsdict[order_id][med_pair_key[0]]["rate"][i]
            volume = medsdict[order_id][med_pair_key[0]]["volume"][i]
            duration = medsdict[order_id][med_pair_key[0]]["duration"][i]
            print(f"  Period {i+1}: {start} to {stop} | Rate: {rate:.1f}mL/h | Volume: {volume:.1f}mL | Duration: {duration:.2f}h")
    
    print(f"{'='*80}\n")
    
    return medsdict


def process_order_multi_med(order_id, order_rows, supertable, medsdict, all_meds_dict):
    """
    Process an order with multiple medications, tracking by (med_name, formulary_name) pairs.
    
    Args:
        order_id: Order ID to process
        order_rows: All rows for this order_id
        supertable: Patient data table
        medsdict: Dictionary for all infusions
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
        return medsdict, all_meds_dict
    
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
            all_meds_dict = add_to_all_meds_dict(row, all_meds_dict)
        print(f"Added to all_meds_dict")
    
    # Filter to infusion pairs only
    remaining_pairs = {k: pairs[k] for k, _ in infusion_pairs}
    
    if len(remaining_pairs) == 0:
        print("\nNo infusion pairs to process")
        return medsdict, all_meds_dict
    
    # Check if all remaining rows are "Not Recorded"
    all_remaining_rows = pd.concat([df for df in remaining_pairs.values()])
    if (all_remaining_rows['formulary_name'] == 'Not Recorded').all():
        print("\nAll remaining infusion rows are 'Not Recorded' - skipping (likely small diluents)")
        return medsdict, all_meds_dict
    
    # Process based on number of infusion pairs
    print(f"\n{len(remaining_pairs)} unique (med_name, formulary_name) combination(s) to process")
    
    if len(remaining_pairs) == 1:
        # Single infusion pair - process normally
        pair_key = list(remaining_pairs.keys())[0]
        print(f"\nProcessing single infusion pair: {pair_key[0]} || {pair_key[1]}")
        
        medsdict[order_id][pair_key[0]] = {
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
        medsdict = process_dilution_pairs(
            order_id, fluid_pair, med_pair, all_remaining_rows, medsdict, supertable
        )
        
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
            # Single medication - check if it's truly an infusion
            print(f"Single medication: {unique_meds[0]}")
            
            # Check if all rows are "Not Recorded" and not is_infusion
            all_not_recorded = (order_rows['med_action'] == 'Not Recorded').all()
            not_is_infusion = not order_rows['is_infusion'].iloc[0]  # Check first row
            
            if all_not_recorded and not_is_infusion:
                # Not truly an infusion - add to all_meds_dict
                print(f"  All rows are 'Not Recorded' and not is_infusion - adding to all_meds_dict")
                for idx, row in order_rows.iterrows():
                    all_meds_dict = add_to_all_meds_dict(row, all_meds_dict)
            else:
                # Process as normal infusion
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
            medsdict[order_id] = {}
            medsdict, all_meds_dict = process_order_multi_med(
                order_id, order_rows, supertable, medsdict, all_meds_dict
            )
            
    return medsdict, all_meds_dict


