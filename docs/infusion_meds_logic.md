MAIN FUNCTION: process_encounter_new(medications_data, patient_data)
    1. Filter to infusion medications only
    2. Remove premix diluents 
    3. Group by order_id
    
    FOR each order_id:
        4. Initialize medication dictionary structure
        5. Get all rows for this order_id
        6. Group by formulary_name (medication type)
        
        FOR each medication:
            7. CALL process_medication_timeline_new()
    
    RETURN medication_dictionary

---

CORE FUNCTION: process_medication_timeline_new(order_id, med_name, med_rows, patient_data, med_dict)
    
    1. Sort medication rows chronologically by med_action_time
    2. Initialize processed_indices = empty_set
    3. Initialize orphaned_infuse_rows = empty_list
    
    FOR each row in chronological order:
        
        IF row already processed:
            SKIP to next row
        
        SWITCH med_action:
        
        CASE "Begin Bag":
            processed_indices.add(current_row)
            
            IF medication not yet set up:
                # First infusion
                params = get_order_params_for_row()
                IF params.final_check == TRUE:
                    actual_duration, used_infuse_indices = calculate_actual_infusion_duration()
                    processed_indices.update(used_infuse_indices)
                    
                    final_duration = actual_duration IF > 0 ELSE theoretical_duration
                    med_stop = last_infuse_time IF available ELSE med_start + duration
                    
                    STORE in med_dict: volume, rate, final_duration, med_start, med_stop
                    SET original_params for fallback
                    med_dict.set = TRUE
                ELSE:
                    SKIP row (invalid parameters)
            ELSE:
                # Subsequent infusion
                params = get_order_params_for_row()
                actual_duration, used_infuse_indices = calculate_actual_infusion_duration()
                processed_indices.update(used_infuse_indices)
                
                volume = params.volume OR original_volume OR previous_volume
                rate = params.rate OR original_rate OR previous_rate  
                duration = actual_duration IF > 0 ELSE calculated_duration
                
                APPEND to med_dict lists
        
        CASE "Rate Change":
            processed_indices.add(current_row)
            
            IF medication not yet set up:
                # Treat as implicit Begin Bag
                CALL handle_implicit_begin_bag()
                CONTINUE
            
            params = get_order_params_for_row()
            new_rate = params.rate OR original_rate
            
            IF rate_change_time < previous_med_stop:
                # During active infusion - split period
                time_elapsed = rate_change_time - previous_med_start
                volume_delivered = previous_rate * time_elapsed
                remaining_volume = previous_volume - volume_delivered
                
                UPDATE previous period: end_time = rate_change_time
                
                actual_duration, used_infuse_indices = calculate_actual_infusion_duration()
                processed_indices.update(used_infuse_indices)
                
                new_duration = actual_duration IF > 0 ELSE remaining_volume/new_rate
                CREATE new period with remaining_volume, new_rate, new_duration
            ELSE:
                # After infusion ended - new infusion
                volume = params.volume OR original_volume
                actual_duration, used_infuse_indices = calculate_actual_infusion_duration()
                processed_indices.update(used_infuse_indices)
                
                duration = actual_duration IF > 0 ELSE theoretical_duration
                CREATE new period with volume, new_rate, duration
        
        CASE "Infuse":
            IF medication not yet set up:
                # Check for upcoming Begin Bag within 1 hour
                upcoming_begin_bag = search_next_hour_for_begin_bag()
                
                IF upcoming_begin_bag found:
                    IGNORE this Infuse (let Begin Bag handle it)
                    processed_indices.add(current_row)
                ELSE:
                    # Treat as implicit Begin Bag
                    CALL handle_implicit_begin_bag()
            ELSE:
                # Check if orphaned (not consumed by Begin Bag/Rate Change)
                IF row not in processed_indices:
                    active_infusion = check_if_within_active_period()
                    
                    IF NOT active_infusion:
                        # Orphaned - treat as implicit Begin Bag
                        CALL handle_implicit_begin_bag()
                    ELSE:
                        processed_indices.add(current_row)
        
        CASE "Not Recorded":
            processed_indices.add(current_row)
            
            IF med_stop is missing:
                IGNORE row (unexpected)
                CONTINUE
            
            med_start = med_action_time
            med_stop = row.med_stop
            actual_duration = med_stop - med_start
            
            params = get_order_params_for_row()
            
            IF params.final_check == TRUE:
                volume = params.volume OR params.rate * actual_duration
                rate = params.rate
                
                IF medication not yet set up:
                    INITIALIZE med_dict with actual times
                ELSE:
                    APPEND to med_dict with actual times
            ELSE:
                IF medication already set up:
                    volume = original_rate * actual_duration
                    rate = original_rate
                    APPEND to med_dict with original params + actual times
                ELSE:
                    IGNORE row
        
        CASE "Bolus":
            processed_indices.add(current_row)
            params = get_order_params_for_row()
            
            IF params.final_check == TRUE:
                # Use calculated parameters normally
                volume = params.volume
                rate = params.rate
                duration = params.duration
            ELSE:
                # Fallback: 1-hour duration
                duration = 1.0
                
                IF params.volume is not NaN:
                    volume = params.volume
                    rate = volume / duration
                ELSE:
                    volume = original_volume OR original_rate * duration
                    rate = volume / duration
            
            med_stop = med_start + duration
            
            IF medication not yet set up:
                INITIALIZE med_dict
            ELSE:
                APPEND to med_dict
        
        CASE other_action:
            IF medication not yet set up AND params.final_check == TRUE:
                CALL handle_implicit_begin_bag()
            ELSE:
                IGNORE row
    
    RETURN updated_med_dict, processed_indices

---

HELPER FUNCTION: handle_implicit_begin_bag(row, med_rows, processed_indices, med_dict)
    
    params = get_order_params_for_row()
    actual_duration, used_infuse_indices = calculate_actual_infusion_duration()
    processed_indices.update(used_infuse_indices)
    
    IF params.final_check == TRUE:
        volume = params.volume OR params.rate * duration
        rate = params.rate  
        duration = actual_duration IF > 0 ELSE params.duration
    ELSE:
        # Fallback to original parameters if medication already set up
        IF medication already set up:
            rate = original_rate
            volume = original_volume OR original_rate * duration
            duration = actual_duration IF > 0 ELSE original_duration
        ELSE:
            SKIP (no fallback available)
    
    med_start = med_action_time
    med_stop = last_infuse_time IF available ELSE med_start + duration
    
    IF medication not yet set up:
        INITIALIZE med_dict and set original_params
    ELSE:
        APPEND to existing med_dict

---

HELPER FUNCTION: calculate_actual_infusion_duration(start_row, med_rows, processed_indices)
    
    start_time = start_row.med_action_time
    start_action = start_row.med_action
    used_infuse_indices = []
    last_infuse_time = None
    
    FOR each subsequent_row after start_row:
        IF subsequent_row not in processed_indices:
            
            IF subsequent_row.med_action == "Infuse":
                last_infuse_time = subsequent_row.med_action_time
                used_infuse_indices.append(subsequent_row.index)
            
            ELIF subsequent_row.med_action == "Begin Bag":
                BREAK (end of this infusion period)
            
            ELIF subsequent_row.med_action == "Rate Change":
                IF start_action == "Rate Change":
                    # Consecutive rate changes - leave for further processing
                    RETURN 0, [], None
                ELSE:
                    BREAK (end of this infusion period)
    
    IF last_infuse_time exists:
        duration = last_infuse_time - start_time
        RETURN duration, used_infuse_indices, last_infuse_time
    ELSE:
        RETURN 0, [], None

---

HELPER FUNCTION: get_order_params_for_row(row, patient_data)
    
    1. Extract parent order parameters (volume, rate, duration from order data)
    2. Extract clinical description parameters (parse text descriptions)
    3. Reconcile parameters using cross-validation logic
    4. Extract infusion parameters (from med columns)
    5. Apply unit conversion if amount_unit != rate_unit
    6. Calculate missing parameters using rate/volume/duration relationships
    7. Validate parameter consistency
    8. Return final parameters with final_check flag

---

FINAL FUNCTION: make_medsdict_to_dataframe(patient_timeline, med_dict)
    
    FOR each medication in med_dict:
        FOR each infusion period:
            IF no errors detected:
                distribute_volume_hourly(med_start, med_stop, rate, patient_timeline)
            ELSE:
                SET error_flag = 1
    
    RETURN hourly_medication_volumes_dataframe