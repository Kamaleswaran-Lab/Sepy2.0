def process_rows(rows_df, medsdict, supertable):
    """
    Process a group of rows that have the same order_id, formulary_name, med_action_time, and med_action.
    If multiple rows exist, apply selection logic to choose the best one.
    
    Args:
        rows_df: DataFrame of rows with same grouping criteria
        medsdict: Medication dictionary to update
        supertable: Patient data table
        
    Returns:
        Updated medsdict
    """
    if len(rows_df) == 1:
        # Single row - proceed as usual
        row = rows_df.iloc[0]
        return process_single_row(row, medsdict, supertable)
    else:
        # Multiple rows - apply selection logic
        print(f"Processing {len(rows_df)} rows with same order_id/formulary_name/med_action_time/med_action")
        
        # Calculate parameters for all rows
        row_params = []
        for idx, row in rows_df.iterrows():
            params = get_order_params_for_row(row, supertable)
            row_params.append({
                'row': row,
                'params': params,
                'idx': idx
            })
        
        # Filter to rows with final_check = True
        valid_rows = [rp for rp in row_params if rp['params']['final_check']]
        
        if not valid_rows:
            print("No rows with final_check=True, using all rows")
            valid_rows = row_params
        
        # Get previous rate if it exists
        prev_rate = None
        row = rows_df.iloc[0]  # Use first row to get order_med_id and med_name
        if (row["order_med_id"] in medsdict and 
            row["med_name"] in medsdict[row["order_med_id"]] and
            len(medsdict[row["order_med_id"]][row["med_name"]]["rate"]) > 0):
            prev_rate = medsdict[row["order_med_id"]][row["med_name"]]["rate"][-1]
        
        # Apply selection logic
        selected_row = select_best_row(valid_rows, prev_rate, rows_df.iloc[0]["med_action"])
        
        if selected_row:
            print(f"Selected row with rate: {selected_row['params']['rate']}")
            return process_single_row(selected_row['row'], medsdict, supertable)
        else:
            print("No valid row selected")
            return medsdict

def select_best_row(row_params, prev_rate, med_action):
    """
    Select the best row from multiple candidates based on the selection criteria.
    
    Args:
        row_params: List of dicts with 'row', 'params', 'idx'
        prev_rate: Previous rate value (can be None)
        med_action: The medication action type
        
    Returns:
        Selected row_param dict or None
    """
    if not row_params:
        return None
    
    # If med_action is "Rate Change", filter out rows where current rate equals previous rate
    if med_action == "Rate Change" and prev_rate is not None:
        filtered_rows = []
        for rp in row_params:
            current_rate = rp['params']['rate']
            if pd.notna(current_rate) and current_rate != prev_rate:
                filtered_rows.append(rp)
        
        if filtered_rows:
            row_params = filtered_rows
            print(f"Filtered to {len(row_params)} rows with rate different from previous ({prev_rate})")
    
    # Filter out rows with rate = 0
    non_zero_rows = []
    for rp in row_params:
        current_rate = rp['params']['rate']
        if pd.notna(current_rate) and current_rate != 0.0:
            non_zero_rows.append(rp)
    
    if non_zero_rows:
        row_params = non_zero_rows
        print(f"Filtered to {len(row_params)} rows with non-zero rates")
    
    # If still multiple rows, average the rates
    if len(row_params) > 1:
        valid_rates = []
        for rp in row_params:
            current_rate = rp['params']['rate']
            if pd.notna(current_rate):
                valid_rates.append(current_rate)
        
        if valid_rates:
            avg_rate = sum(valid_rates) / len(valid_rates)
            print(f"Averaging {len(valid_rates)} rates: {valid_rates} -> {avg_rate:.2f}")
            
            # Use the first row but with averaged rate
            best_row = row_params[0].copy()
            best_row['params'] = best_row['params'].copy()
            best_row['params']['rate'] = avg_rate
            
            # Recalculate duration and med_stop based on averaged rate
            if pd.notna(best_row['params']['volume']) and avg_rate > 0:
                best_row['params']['duration'] = best_row['params']['volume'] / avg_rate
                best_row['params']['med_stop'] = best_row['params']['med_start'] + pd.Timedelta(hours=best_row['params']['duration'])
            
            return best_row
    
    # Default to first row if no clear winner
    if row_params:
        print("Defaulting to first available row")
        return row_params[0]
    
    return None

def process_single_row(row, medsdict, supertable):
    if not medsdict[row["order_med_id"]][row["med_name"]]["set"]:
        print("This med doesnt have params set yet")
        params = get_order_params_for_row(row, supertable)
        if params["final_check"]:
            print("Setting unset params")
            medsdict[row["order_med_id"]][row["med_name"]]["volume"] = [params["volume"]]
            medsdict[row["order_med_id"]][row["med_name"]]["duration"] = [params["duration"]]
            medsdict[row["order_med_id"]][row["med_name"]]["rate"] = [params["rate"]]
            medsdict[row["order_med_id"]][row["med_name"]]["med_start"] = [params["med_start"]]
            medsdict[row["order_med_id"]][row["med_name"]]["med_stop"] = [params["med_stop"]]
            medsdict[row["order_med_id"]][row["med_name"]]["set"] = True
            
            # Store original parameters for fallback use
            medsdict[row["order_med_id"]][row["med_name"]]["original_rate"] = params["rate"]
            medsdict[row["order_med_id"]][row["med_name"]]["original_volume"] = params["volume"]
            medsdict[row["order_med_id"]][row["med_name"]]["original_duration"] = params["duration"]
            print(f"Stored original params: rate={params['rate']}, volume={params['volume']}, duration={params['duration']}")
        else:
            print("Final check failed for initial params - skipping this row")
            # If no original params stored yet, we can't do much
            pass
    elif row["med_action"] in ["Begin Bag", "Infuse"]:
        action = row["med_action"]
        print(f"Med action is {action}") 
        med_action_time = pd.to_datetime(row["med_action_time"])
        med_stop_prev = medsdict[row["order_med_id"]][row["med_name"]]["med_stop"][-1]
        
        if (action == "Infuse") and (med_action_time < med_stop_prev):
            print("Infuse is during active infusion, skipping")
            return medsdict

        params = get_order_params_for_row(row, supertable)
        prev_volume = medsdict[row["order_med_id"]][row["med_name"]]["volume"][-1]
        prev_rate = medsdict[row["order_med_id"]][row["med_name"]]["rate"][-1]
        prev_duration = medsdict[row["order_med_id"]][row["med_name"]]["duration"][-1]

        original_rate = medsdict[row["order_med_id"]][row["med_name"]]["original_rate"]
        original_volume = medsdict[row["order_med_id"]][row["med_name"]]["original_volume"]
        original_duration = medsdict[row["order_med_id"]][row["med_name"]]["original_duration"]

        if params["final_check"]:
            print("Setting new params")
            new_volume = params["volume"]
            new_rate = params["rate"]
            new_duration = params["duration"]
            medsdict[row["order_med_id"]][row["med_name"]]["med_start"].append(params["med_start"])
            if pd.notna(new_volume):
                medsdict[row["order_med_id"]][row["med_name"]]["volume"].append(new_volume)
            elif pd.notna(original_volume):
                print("Using original volume") 
                medsdict[row["order_med_id"]][row["med_name"]]["volume"].append(original_volume)
            else:
                print("Using previous volume") 
                medsdict[row["order_med_id"]][row["med_name"]]["volume"].append(prev_volume)

            if pd.notna(new_rate):
                medsdict[row["order_med_id"]][row["med_name"]]["rate"].append(new_rate)
            elif pd.notna(original_rate):
                print("Using original rate")
                medsdict[row["order_med_id"]][row["med_name"]]["rate"].append(original_rate)
            else:
                print("Using previous rate")
                medsdict[row["order_med_id"]][row["med_name"]]["rate"].append(prev_rate)

            if pd.notna(new_duration):
                print("Using new duration")
                medsdict[row["order_med_id"]][row["med_name"]]["duration"].append(new_duration)
                medsdict[row["order_med_id"]][row["med_name"]]["med_stop"].append(params["med_start"] + pd.Timedelta(hours = new_duration))
            elif pd.notna(original_duration):
                print("Using original duration")
                medsdict[row["order_med_id"]][row["med_name"]]["duration"].append(original_duration)
                medsdict[row["order_med_id"]][row["med_name"]]["med_stop"].append(params["med_start"] + pd.Timedelta(hours = original_duration))
            else:
                print("Using previous duration")
                medsdict[row["order_med_id"]][row["med_name"]]["duration"].append(prev_duration)
                medsdict[row["order_med_id"]][row["med_name"]]["med_stop"].append(params["med_start"] + pd.Timedelta(hours = prev_duration))
        else:
            print("Final check failed - using original parameters as fallback")
            # Use original parameters as fallback
            original_rate = medsdict[row["order_med_id"]][row["med_name"]]["original_rate"]
            original_volume = medsdict[row["order_med_id"]][row["med_name"]]["original_volume"]
            original_duration = medsdict[row["order_med_id"]][row["med_name"]]["original_duration"]
            
            if pd.notna(original_rate) and pd.notna(original_duration):
                print(f"Using original: rate={original_rate}, duration={original_duration}")
                #medsdict[row["order_med_id"]][row["med_name"]]["volume"].append(original_volume)
                medsdict[row["order_med_id"]][row["med_name"]]["rate"].append(original_rate)
                medsdict[row["order_med_id"]][row["med_name"]]["duration"].append(original_duration)
                medsdict[row["order_med_id"]][row["med_name"]]["med_start"].append(params["med_start"])
                medsdict[row["order_med_id"]][row["med_name"]]["med_stop"].append(params["med_start"] + pd.Timedelta(hours = original_duration))
            else:
                print("No original parameters, skipping this row")
                return medsdict
                
        if (med_action_time >= med_stop_prev):
            print("Med is starting after previous one finished") 
        else:
            print("Overlapping periods - stopping prev med")
            medsdict[row["order_med_id"]][row["med_name"]]["med_stop"][-2] = med_action_time

    elif row["med_action"] == "Rate Change":
        med_action_time = pd.to_datetime(row["med_action_time"])
        params = get_order_params_for_row(row, supertable)
        original_rate = medsdict[row["order_med_id"]][row["med_name"]]["original_rate"]
        original_volume = medsdict[row["order_med_id"]][row["med_name"]]["original_volume"]
        original_duration = medsdict[row["order_med_id"]][row["med_name"]]["original_duration"]
        
        if params["final_check"]:
            new_volume = params["volume"]
            new_rate = params["rate"]
            new_duration = params["duration"]
            prev_volume = medsdict[row["order_med_id"]][row["med_name"]]["volume"][-1]
            prev_rate = medsdict[row["order_med_id"]][row["med_name"]]["rate"][-1]
            prev_duration = medsdict[row["order_med_id"]][row["med_name"]]["duration"][-1]
            if pd.isna(prev_volume) and ~pd.isna(prev_rate) and ~pd.isna(prev_duration):
                prev_volume = prev_duration*prev_rate

            prev_start = medsdict[row["order_med_id"]][row["med_name"]]["med_start"][-1]
            prev_stop = medsdict[row["order_med_id"]][row["med_name"]]["med_stop"][-1]
            
            # Check if rate change occurs during active infusion
            if med_action_time < prev_stop:
                # First check if there are actual parameter changes
                if (new_rate == prev_rate):
                    print("Rate change during infusion but no actual parameter changes - ignoring")
                    return medsdict
                elif not pd.isna(new_rate) and (new_rate > 0):
                    print("Rate change during active infusion - splitting periods")
                    
                    # Calculate volume already delivered before rate change
                    time_elapsed = (med_action_time - prev_start).total_seconds() / 3600.0
                    volume_delivered = prev_rate * time_elapsed
                    remaining_volume = prev_volume - volume_delivered
                    
                    # Update the previous period to end at rate change time
                    medsdict[row["order_med_id"]][row["med_name"]]["med_stop"][-1] = med_action_time
                    medsdict[row["order_med_id"]][row["med_name"]]["duration"][-1] = time_elapsed
                    
                    print(f"Previous period: delivered {volume_delivered:.1f}mL in {time_elapsed:.2f}hrs")
                    print(f"Remaining volume: {remaining_volume:.1f}mL")
                    
                    # Start new period with remaining volume and new rate
                    
                    new_period_duration = remaining_volume / new_rate
                    new_period_stop = med_action_time + pd.Timedelta(hours=new_period_duration)
                    
                    medsdict[row["order_med_id"]][row["med_name"]]["volume"].append(remaining_volume)
                    medsdict[row["order_med_id"]][row["med_name"]]["rate"].append(new_rate)
                    medsdict[row["order_med_id"]][row["med_name"]]["duration"].append(new_period_duration) 
                    medsdict[row["order_med_id"]][row["med_name"]]["med_start"].append(med_action_time)
                    medsdict[row["order_med_id"]][row["med_name"]]["med_stop"].append(new_period_stop)
                    
                    print(f"New period: {remaining_volume:.1f}mL at {new_rate:.1f}mL/hr for {new_period_duration:.2f}hrs")
                else:
                    print("New rate is no valid - ignoring")
                    return medsdict
                    
            elif med_action_time >= prev_stop:
                print("Rate change after infusion ended - starting new period")
                if pd.notna(new_volume):
                    print("Using new volume")
                    medsdict[row["order_med_id"]][row["med_name"]]["volume"].append(new_volume)
                elif pd.notna(original_volume):
                    print("Using original volume")
                    medsdict[row["order_med_id"]][row["med_name"]]["volume"].append(original_volume)
                else:
                    print("Using previous volume")
                    medsdict[row["order_med_id"]][row["med_name"]]["volume"].append(prev_volume)

                if pd.notna(new_rate):
                    print("Using new rate")
                    medsdict[row["order_med_id"]][row["med_name"]]["rate"].append(new_rate)
                elif pd.notna(original_rate):
                    print("Using original rate")
                    medsdict[row["order_med_id"]][row["med_name"]]["rate"].append(original_rate)
                else:
                    print("Using previous rate")
                    medsdict[row["order_med_id"]][row["med_name"]]["rate"].append(prev_rate)

                if pd.notna(new_duration):
                    print("Using new duration")
                    medsdict[row["order_med_id"]][row["med_name"]]["duration"].append(new_duration)
                elif pd.notna(original_duration):
                    print("Using original duration")
                    medsdict[row["order_med_id"]][row["med_name"]]["duration"].append(original_duration)
                else:
                    print("Using previous duration")
                    medsdict[row["order_med_id"]][row["med_name"]]["duration"].append(prev_duration)

                medsdict[row["order_med_id"]][row["med_name"]]["med_start"].append(med_action_time)
                medsdict[row["order_med_id"]][row["med_name"]]["med_stop"].append(med_action_time + pd.Timedelta(hours=new_duration))
        else:
            print("Final check failed for a rate change row - ignoring the row")

    return medsdict

    def calculate_infusion_duration(row, meds, ongoing_infusion):
    """
    Calculate infusion duration for "Begin Bag" actions by finding subsequent actions.
    
    Args:
        row: Current medication row
        meds: Complete medications DataFrame
        ongoing_infusion: Dict tracking ongoing infusions
        
    Returns:
        float: Duration in hours, or None if cannot be calculated
    """
    
    # Get all rows for this medication
    meds_slice = meds.loc[
        (meds["order_med_id"] == row["order_med_id"]) & 
        (meds["formulary_name"] == row["formulary_name"])
    ].sort_values("med_action_time")
    
    if len(meds_slice) == 1:
        return None
    
    # Track this infusion
    ongoing_infusion[row["order_med_id"]] = row["med_action_time"]
    start_time = row["med_action_time"]
    
    # Find subsequent rows for this medication
    current_idx = row.name
    subsequent_rows = meds_slice[meds_slice.index > current_idx].sort_values('med_action_time')
    
    end_time = None
    last_infuse_time = None
    
    for _, next_row in subsequent_rows.iterrows():
        if next_row["med_action"] == "Infuse":
            last_infuse_time = next_row["med_action_time"]
        elif (next_row["med_action"] == "Begin Bag") or (next_row["med_action"] == "Rate Change"):
            end_time = last_infuse_time
            break
    
    if end_time:
        # Calculate duration to nearest whole hour
        raw_duration = pd.to_datetime(end_time) - pd.to_datetime(start_time)
        duration_hours = round(raw_duration.total_seconds() / 3600.0)
        return duration_hours
    
    return None