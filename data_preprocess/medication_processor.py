from pathlib import Path
import pandas as pd
import numpy as np
import sys
sys.path.append("../")
import utils

### NOT RECORDED are the ones with med stop!!!!

def check_duration_in_desc(row):
    """
    Check if duration from parent order matches description text.
    
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
    Extract infusion parameters from parent order data.
    
    Args:
        row: DataFrame row with parent order information
        
    Returns:
        dict: Contains volume, rate, duration, and suspicious flag
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
    Extract infusion parameters from clinical description text.
    
    Args:
        row: DataFrame row with ORDER_CLINICAL_DESC
        
    Returns:
        dict: Contains volume, rate, duration, and sus flag
    """
    if not row["ORDER_CLINICAL_DESC"] or pd.isna(row["ORDER_CLINICAL_DESC"]):
        return {"has_clinical_desc": False, "sus": False}
    
    params = utils.parse_clinical_description(row["ORDER_CLINICAL_DESC"])
    sus = False
    
    # Helper function to check if value is valid (not None and not NaN)
    def is_valid_param(param):
        if param is None:
            return False
        if isinstance(param, list):
            return len(param) > 0
        return not pd.isna(param)
    
    # Handle multiple volumes
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
    
    # Handle multiple rates
    if params["rate"]:
        if len(params["rate"]) > 1:
            sus = True
        else:
            params["rate"] = params["rate"][0]
    else:
        params["rate"] = None
    
    # Handle multiple durations
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


def calculate_dose_based_rate(row, weight):
    """
    Calculate infusion rate from medication action dose.
    
    Args:
        row: DataFrame row with med_action_dose and med_action_dose_unit
        weight: Patient weight in kg
        
    Returns:
        dict: Contains rate, rate_unit, duration
    """
    if pd.isna(row["med_action_dose"]):
        return {"rate": None, "rate_unit": None, "duration": None}
    
    med_action_dose = row["med_action_dose"]
    med_action_dose_unit = row["med_action_dose_unit"]
    
    rate = None
    rate_unit = None
    duration = None
    
    # Convert various dose units to rates
    if med_action_dose_unit == 'Not Recorded':
        rate = None
    elif med_action_dose_unit == 'Milligrams/Minute':
        rate = med_action_dose * 60
        rate_unit = 'Milligrams/Hour'
    elif med_action_dose_unit == 'Micrograms/Hour':
        rate = med_action_dose
        rate_unit = 'Micrograms/Hour'
    elif med_action_dose_unit == 'Microgram/Kilogram/Minute':
        rate = med_action_dose * weight * 60
        rate_unit = 'Micrograms/Hour'
    elif med_action_dose_unit == 'Microgram/Kilogram/Hour':
        rate = med_action_dose * weight
        rate_unit = 'Micrograms/Hour'
    elif med_action_dose_unit == 'Milligram/Hour':
        rate = med_action_dose
        rate_unit = 'Milligrams/Hour'
    elif med_action_dose_unit == 'Milliequivalents/Minute':
        rate = med_action_dose * 60
        rate_unit = 'Milliequivalents/Hour'
    elif med_action_dose_unit == 'ng/kg/min':
        rate = med_action_dose * weight * 60
        rate_unit = 'Nanograms/Hour'
    elif med_action_dose_unit == 'Milligram/Kilogram/Hour':
        rate = med_action_dose * weight
        rate_unit = 'Milligrams/Hour'
    elif med_action_dose_unit == 'U/Hr':
        rate = med_action_dose
        rate_unit = 'Units/Hour'
    elif med_action_dose_unit == 'Micrograms/Minute':
        rate = med_action_dose * 60
        rate_unit = 'Micrograms/Hour'
    elif med_action_dose_unit == 'Unit/Minute':
        rate = med_action_dose * 60
        rate_unit = 'Units/Hour'
    elif med_action_dose_unit == 'Grams/Hour':
        rate = med_action_dose
        rate_unit = 'Grams/Hour'
    elif med_action_dose_unit == 'Unit/Kilogram/Hour':
        rate = med_action_dose * weight
        rate_unit = 'Units/Hour'
    elif med_action_dose_unit == 'Milligram/Kilogram/Minute':
        rate = med_action_dose * weight * 60
        rate_unit = 'Milligrams/Hour'
    elif med_action_dose_unit == 'Milliequivalents/Kilogram/Hour':
        rate = med_action_dose * weight
        rate_unit = 'Milliequivalents/Hour'
    elif med_action_dose_unit == 'Milligrams/Millilit':
        rate = med_action_dose
        rate_unit = 'Milligrams/Milliliter'
    elif med_action_dose_unit == 'minute(s)':
        duration = med_action_dose / 60.0  # Convert to hours
    elif med_action_dose_unit == 'Hour':
        duration = med_action_dose  # Already in hours
    
    return {"rate": rate, "rate_unit": rate_unit, "duration": duration}


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


def calculate_actual_infusion_duration(start_row, all_med_rows, processed_indices):
    """
    NEW FUNCTION: Calculate actual infusion duration based on "Infuse" actions.
    
    Args:
        start_row: Row with "Begin Bag" or "Rate Change" action that starts the infusion
        all_med_rows: All rows for this order_med_id + formulary_name, sorted by time
        processed_indices: Set of row indices that have already been processed
        
    Returns:
        tuple: (duration_hours, list_of_used_infuse_indices, last_infuse_time)
    """
    start_time = pd.to_datetime(start_row["med_action_time"])
    start_idx = start_row.name
    start_action = start_row["med_action"]
    
    # Find subsequent rows for this medication that haven't been processed
    subsequent_rows = all_med_rows[
        (all_med_rows.index > start_idx) & 
        (~all_med_rows.index.isin(processed_indices))
    ].sort_values('med_action_time')
    
    last_infuse_time = None
    used_infuse_indices = []
    
    for _, next_row in subsequent_rows.iterrows():
        if next_row["med_action"] == "Infuse":
            last_infuse_time = pd.to_datetime(next_row["med_action_time"])
            used_infuse_indices.append(next_row.name)
        elif (next_row["med_action"] == "Begin Bag") or (next_row["med_action"] == "Waste"):
            # Stop when we hit the next bag - this ends the current infusion period
            break
        elif next_row["med_action"] == "Rate Change":
            # Hitting a "Rate Change" means 
            # we should leave it for further processing - don't consume any infuse actions
            print(f"Rate Change followed by another Rate Change - leaving for further processing")
            return 0.0, [], None
    
    if last_infuse_time:
        # Calculate actual duration based on last infuse time
        raw_duration = last_infuse_time - start_time
        duration_hours = raw_duration.total_seconds() / 3600.0
        return duration_hours, used_infuse_indices, last_infuse_time
    else:
        # No infuse actions found - infusion may have been stopped immediately
        action_desc = "Rate Change" if start_action == "Rate Change" else "Begin Bag"
        print(f"WARNING: No 'Infuse' actions found after '{action_desc}' at {start_time}")
        return 0.0, [], None


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

def process_premix(imeds):
    """
    Process using with "Premix Diluent" as the formulary name
    
    """
    premix_solutions = [ "Premix Diluent", "Premix NS", "Premix Dextrose 5%", 
                        "Premix Water, Sterile" ]
    premix = imeds.loc[imeds.formulary_name.isin(premix_solutions)]
    imeds = imeds.loc[~imeds.formulary_name.isin(premix_solutions)]

    # Create a set of (order_med_id, med_action_time) tuples from imeds for faster lookup
    imeds_pairs = set(zip(imeds['order_med_id'], imeds['med_action_time']))

    # Check if each premix row has a matching pair in imeds
    premix.loc[:, "checked"] = [
        (order_id, action_time) in imeds_pairs 
        for order_id, action_time in zip(premix['order_med_id'], premix['med_action_time'])
    ]
    
    if not premix["checked"].all():
        print("Some premix are not matched")
    
    return imeds 

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

    amount_unit_mapping = {
        'mg': 'Milligram',
        'unit': 'Unit', 
        'gm': 'Gram',
        'g': 'Gram',
        'mcg': 'Microgram',
        'mEq': 'Milliequivalent',
        'ng': 'ng',  
        '%': '%',    
        'mmol': 'mmol',  
    }
    
    # Unit conversion factors (to convert FROM unit TO base unit)
    unit_conversion_factors = {
        # Weight conversions (to milligrams)
        'Gram': 1000,
        'Milligram': 1,
        'Microgram': 0.001,
        'ng': 0.000001,
        # Volume conversions (to milliliters) 
        'Liter': 1000,
        'Milliliter': 1,
        # Other units (no conversion)
        'Unit': 1,
        'Milliequivalent': 1,
        '%': 1,
        'mmol': 1
    }
    
    def convert_units(amount, from_unit, to_unit):
        """Convert amount from one unit to another if possible."""
        if from_unit == to_unit:
            return amount
            
        # Get conversion factors
        from_factor = unit_conversion_factors.get(from_unit)
        to_factor = unit_conversion_factors.get(to_unit)
        
        if from_factor is None or to_factor is None:
            return None  # Cannot convert
            
        # Convert: amount * (from_factor / to_factor)
        return amount * (from_factor / to_factor)

    if not pd.isna(row["med_start"]):
        med_start = pd.to_datetime(row["med_start"])
    
        if not pd.isna(row["med_stop"]):
            med_stop = pd.to_datetime(row["med_stop"])
            duration = (med_stop - med_start).total_seconds() / 3600.0
    
    if row["is_infusion"]:
        if row["volume_inf"] is not None:
            volume = row["volume_inf"]*1000 if row["volume_inf_unit"] == "L" else row["volume_inf"]
            volume_unit = row["volume_inf_unit"]
        if row["amount_inf"] is not None:
            amount = row["amount_inf"]
            amount_unit = row["amount_inf_unit"] 
            amount_unit_mapped = amount_unit_mapping[amount_unit] if amount_unit in amount_unit_mapping else amount_unit

        if row["med_action_dose"] is not None:
            results = calculate_dose_based_rate(row, weight)
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
    

def add_med_to_final_df(medsdf, formulary_name):
    if formulary_name not in medsdf.columns:
        medsdf[formulary_name] = 0.0
    
    return medsdf

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

def get_order_params_for_row(row, supertable):
    """
    Get volume, rate, duration for a row

    """
    nearest_idx = supertable.index.get_indexer([pd.to_datetime(row["med_action_time"])], method='nearest')[0]
    weight = supertable['daily_weight_kg'].values[nearest_idx]
    parent_order_params = extract_parent_order_params(row)
    clinical_desc_params = extract_clinical_desc_params(row)
    reconciled_params = reconcile_parameters(parent_order_params, clinical_desc_params)
    infusion_params = extract_infusion_params(row, weight)
    med_start = pd.to_datetime(row["med_action_time"])

    volume = reconciled_params["volume"]
    rate = reconciled_params["rate"]
    duration = reconciled_params["duration"]
    final_check = reconciled_params["final_check"]

    for key in infusion_params.keys():
        if isinstance(infusion_params[key], float) and np.isnan(infusion_params[key]):
            infusion_params[key] = None
    
    if (pd.notna(infusion_params["med_stop"])) and \
          (infusion_params["duration_inf"] != 0.0):
        duration_inf = (infusion_params["med_stop"] - infusion_params["med_start"]).total_seconds() / 3600.0
        if pd.notna(duration) and duration != 0.0:
            if duration_inf != duration:
                duration = duration_inf 
                print("Duration mismatch - using infusion duration")
            else:
                print("Duration match - using reconciled duration")
        else:
            print("Using infusion duration")
            duration = duration_inf

    if (not final_check) or (pd.isna(rate)):
        print("Getting infusion params")
        if (infusion_params["rate_unit"] is not None) and (infusion_params["amount_inf"] is not None):
            # Check if units match directly or can be converted
            units_compatible = False
            converted_amount = infusion_params["amount_inf"]
            
            if infusion_params["amount_inf_unit_mapped"].lower() in infusion_params["rate_unit"].lower():
                # Direct unit match (existing logic)
                units_compatible = True
                print(f"Direct unit match: {infusion_params['amount_inf_unit_mapped']} in {infusion_params['rate_unit']}")
            else:
                # Try unit conversion
                # Extract base unit from rate_unit (e.g., "Milligrams/Hour" -> "Milligrams")
                rate_base_unit = infusion_params["rate_unit"].split('/')[0] if '/' in infusion_params["rate_unit"] else infusion_params["rate_unit"]
                
                # Define unit conversion helper inside the function scope
                unit_conversion_factors = {
                    'Gram': 1000, 'Milligram': 1, 'Microgram': 0.001, 'ng': 0.000001,
                    'Liter': 1000, 'Milliliter': 1,
                    'Unit': 1, 'Milliequivalent': 1, '%': 1, 'mmol': 1
                }
                
                def convert_units_local(amount, from_unit, to_unit):
                    if from_unit == to_unit:
                        return amount
                    def normalize_unit(unit):
                        if unit.endswith('s'):
                            unit = unit[:-1]
                        return unit
                    from_factor = unit_conversion_factors.get(normalize_unit(from_unit))
                    to_factor = unit_conversion_factors.get(normalize_unit(to_unit))
                    if from_factor is None or to_factor is None:
                        return None
                    return amount * (from_factor / to_factor)
                
                converted_amount = convert_units_local(
                    infusion_params["amount_inf"],
                    infusion_params["amount_inf_unit_mapped"],
                    rate_base_unit
                )
                
                if converted_amount is not None:
                    units_compatible = True
                    print(f"Unit conversion successful: {infusion_params['amount_inf']} {infusion_params['amount_inf_unit_mapped']} -> {converted_amount:.3f} {rate_base_unit}")
                else:
                    print(f"Cannot convert {infusion_params['amount_inf_unit_mapped']} to {rate_base_unit}")
            
            if units_compatible:
                if infusion_params["rate_inf"] == 0.0:
                    rate = 0.0
                    duration = 0.0
                    volume = 0.0
                else:    
                    duration = converted_amount / infusion_params["rate_inf"]
                    if not pd.isna(infusion_params["volume_inf"]):
                        rate = infusion_params["volume_inf"] / duration
                        final_check = True
                    else:
                        final_check = False 

                if (pd.notna(infusion_params["med_stop"])) & (duration != infusion_params["duration_inf"]): #Maybe bag didnt end 
                    duration = infusion_params["duration_inf"]
                    volume = rate*duration  
            else:
                final_check = False
        elif infusion_params["volume_inf"] is not None:
            volume = infusion_params["volume_inf"]
            if duration is None or pd.isna(duration) or duration == 0.0:
                duration = infusion_params["duration_inf"]
                if duration is None or pd.isna(duration) or duration == 0.0:
                    if (volume <= 1000) and (infusion_params["is_fluid"]):
                        duration = 1
                        rate = volume
                        final_check = True
                    else:
                        final_check = False
                else:
                    rate = volume/duration
                    final_check = True
            else:
                rate = volume/duration
                final_check = True
        else:
            final_check = False 
    #else:
    #    print("Using reconciled params")
    #    final_check = True
        
    if (pd.notna(infusion_params["med_stop"])):
        med_stop = infusion_params["med_stop"] 
    elif pd.notna(duration) and duration != 0.0: 
        med_stop = med_start + pd.Timedelta(hours=duration)
    else:
        med_stop = np.nan 
    
    print(f"Med start: {med_start}, Med stop: {med_stop}, Volume: {volume}, Rate: {rate}, Duration: {duration}, Final Check: {final_check}")
    return {
        "volume": volume,
        "rate": rate,
        "duration": duration,
        "final_check": final_check,
        "med_start": med_start,
        "med_stop": med_stop
    }


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

def make_medsdict_to_dataframe(supertable: pd.DataFrame, medsdict: dict):
    medsdf = pd.DataFrame(index = supertable.index)
    
    meds = []
    for order_id, med_dict in medsdict.items():
        for med, med_data in med_dict.items():
            if med not in meds:
                meds.append(med)
    
    # Create medication columns (volumes)
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


def handle_implicit_begin_bag(row, med_rows, processed_indices, order_id, med_name, medsdict, supertable):
    """
    Handle case where first action is not Begin Bag - create implicit infusion period.
    
    Args:
        row: Current row (Infuse or Rate Change)
        med_rows: All rows for this medication
        processed_indices: Set of already processed indices
        order_id: Order ID
        med_name: Medication name
        medsdict: Medication dictionary
        supertable: Patient data table
    
    Returns:
        tuple: (updated_medsdict, updated_processed_indices)
    """
    print(f"Creating implicit Begin Bag for first {row['med_action']} action")
    
    # Get theoretical parameters for this action
    params = get_order_params_for_row(row, supertable)
    
    # Calculate actual duration from Infuse actions
    actual_duration, used_infuse_indices, last_infuse_time = calculate_actual_infusion_duration(
        row, med_rows, processed_indices
    )
    processed_indices.update(used_infuse_indices)
    
    # Determine final parameters
    if params["final_check"]:
        rate = params["rate"]
        volume = params["volume"]
        duration = actual_duration if actual_duration > 0 else params["duration"]
        
        # Calculate volume if missing
        if pd.isna(volume) and pd.notna(rate) and pd.notna(duration):
            volume = rate * duration
            
        print(f"Using calculated params: Rate={rate:.1f}mL/h, Volume={volume:.1f}mL")
    else:
        print("Final check failed for implicit Begin Bag - checking for original parameters")
        
        # Check if medication is already set up and use original parameters
        if medsdict[order_id][med_name]["set"]:
            rate = medsdict[order_id][med_name]["original_rate"]
            volume = medsdict[order_id][med_name]["original_volume"]
            original_duration = medsdict[order_id][med_name]["original_duration"]
            duration = actual_duration if actual_duration > 0 else original_duration
            
            # Calculate volume if missing using original rate and duration
            if pd.isna(volume) and pd.notna(rate) and pd.notna(duration):
                volume = rate * duration
                
            print(f"Using original params: Rate={rate:.1f}mL/h, Volume={volume:.1f}mL, Duration={duration:.2f}h")
        else:
            print("No original parameters available - skipping this medication")
            return medsdict, processed_indices
    
    # Set med_start and med_stop
    med_start = pd.to_datetime(row["med_action_time"])
    if last_infuse_time:
        med_stop = last_infuse_time
        # Recalculate duration based on actual times
        duration = (med_stop - med_start).total_seconds() / 3600.0
        print(f"Using actual duration from Infuse actions: {duration:.2f}h")
    else:
        med_stop = med_start + pd.Timedelta(hours=duration)
        print(f"Using theoretical duration: {duration:.2f}h")
    
    # Check if medication is already set up, then append or initialize
    if not medsdict[order_id][med_name]["set"]:
        # Initialize the medication in medsdict (first time)
        medsdict[order_id][med_name]["volume"] = [volume]
        medsdict[order_id][med_name]["rate"] = [rate]
        medsdict[order_id][med_name]["duration"] = [duration]
        medsdict[order_id][med_name]["med_start"] = [med_start]
        medsdict[order_id][med_name]["med_stop"] = [med_stop]
        medsdict[order_id][med_name]["set"] = True
        
        # Store original parameters for fallback use (only on first setup)
        medsdict[order_id][med_name]["original_rate"] = rate
        medsdict[order_id][med_name]["original_volume"] = volume
        medsdict[order_id][med_name]["original_duration"] = duration
        
        print(f"IMPLICIT BEGIN BAG (INITIAL) - Final params: Start={med_start}, Stop={med_stop}, Rate={rate:.1f}mL/h, Volume={volume:.1f}mL, Duration={duration:.2f}h")
    else:
        # Append to existing medication (subsequent implicit begin bag)
        medsdict[order_id][med_name]["volume"].append(volume)
        medsdict[order_id][med_name]["rate"].append(rate)
        medsdict[order_id][med_name]["duration"].append(duration)
        medsdict[order_id][med_name]["med_start"].append(med_start)
        medsdict[order_id][med_name]["med_stop"].append(med_stop)
        
        print(f"IMPLICIT BEGIN BAG (SUBSEQUENT) - Final params: Start={med_start}, Stop={med_stop}, Rate={rate:.1f}mL/h, Volume={volume:.1f}mL, Duration={duration:.2f}h")
    
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
        
    Returns:
        tuple: (updated_medsdict, processed_row_indices)
    """
    # Get rows for this specific medication
    med_rows = all_order_rows[all_order_rows['formulary_name'] == med_name].sort_values('med_action_time')
    processed_indices = set()
    orphaned_infuse_rows = []
    
    print(f"Processing {med_name} with {len(med_rows)} rows")
    
    for _, row in med_rows.iterrows():
        if row.name in processed_indices:
            continue
            
        med_action = row["med_action"]
        med_action_time = pd.to_datetime(row["med_action_time"])
        
        print(f"Processing {med_action} at {med_action_time}")
        
        if med_action == "Begin Bag":
            processed_indices.add(row.name)
            
            if not medsdict[order_id][med_name]["set"]:
                # First infusion - establish original parameters
                params = get_order_params_for_row(row, supertable)
                if params["final_check"]:
                    print("Setting initial params from Begin Bag")
                    
                    # Calculate actual duration based on Infuse actions
                    actual_duration, used_infuse_indices, last_infuse_time = calculate_actual_infusion_duration(
                        row, med_rows, processed_indices
                    )
                    
                    # Mark infuse rows as processed
                    processed_indices.update(used_infuse_indices)
                    
                    # Use actual duration if it's different from theoretical
                    if actual_duration > 0:
                        final_duration = actual_duration
                        med_stop = last_infuse_time
                        print(f"Using actual duration: {final_duration:.2f} hours (vs theoretical: {params['duration']})")
                    else:
                        final_duration = params["duration"] if pd.notna(params["duration"]) else 0
                        med_stop = params["med_stop"]
                        print(f"Using theoretical duration: {final_duration} hours")
                    
                    # Store parameters
                    if pd.isna(params["volume"]):
                        params["volume"] = params["rate"]*params["duration"]

                    medsdict[order_id][med_name]["volume"] = [params["volume"]]
                    medsdict[order_id][med_name]["rate"] = [params["rate"]]
                    medsdict[order_id][med_name]["duration"] = [final_duration]
                    medsdict[order_id][med_name]["med_start"] = [params["med_start"]]
                    medsdict[order_id][med_name]["med_stop"] = [med_stop]
                    medsdict[order_id][med_name]["set"] = True
                    
                    # Store originals for fallback
                    medsdict[order_id][med_name]["original_rate"] = params["rate"]
                    medsdict[order_id][med_name]["original_volume"] = params["volume"]
                    medsdict[order_id][med_name]["original_duration"] = params["duration"]
                    
                    print(f"INITIAL BEGIN BAG - Final params: Start={params['med_start']}, Stop={med_stop}, Rate={params['rate']:.1f}mL/h, Volume={params['volume']:.1f}mL, Duration={final_duration:.2f}h")
                else:
                    print("Final check failed for initial Begin Bag - skipping")
            else:
                # Subsequent infusion bag
                print("Processing subsequent Begin Bag")
                params = get_order_params_for_row(row, supertable)
                
                # Calculate actual duration for this new bag
                actual_duration, used_infuse_indices, last_infuse_time = calculate_actual_infusion_duration(
                    row, med_rows, processed_indices
                )
                processed_indices.update(used_infuse_indices)
                
                # Use original parameters if current ones fail validation
                if params["final_check"]:
                    volume = params["volume"]
                    rate = params["rate"] 
                    duration = actual_duration if actual_duration > 0 else params["duration"]
                    if pd.isna(volume):
                        volume = rate*duration 
                    print(f"Beginning bag with volume : {volume}, rate : {rate}, duration : {duration}")
                else:
                    volume = medsdict[order_id][med_name]["original_volume"]
                    rate = medsdict[order_id][med_name]["original_rate"]
                    duration = actual_duration if actual_duration > 0 else medsdict[order_id][med_name]["original_duration"]
                    print("Using original parameters as fallback")
                
                # Add new infusion period
                medsdict[order_id][med_name]["volume"].append(volume)
                medsdict[order_id][med_name]["rate"].append(rate)
                medsdict[order_id][med_name]["duration"].append(duration)
                medsdict[order_id][med_name]["med_start"].append(params["med_start"])

                if (last_infuse_time is not None) and (last_infuse_time - params["med_start"] > pd.Timedelta(hours=0)):
                    medsdict[order_id][med_name]["med_stop"].append(last_infuse_time)
                    final_stop = last_infuse_time
                else:
                    final_stop = params["med_start"] + pd.Timedelta(hours=duration)
                    medsdict[order_id][med_name]["med_stop"].append(final_stop)
                print(medsdict[order_id][med_name])
                print(f"SUBSEQUENT BEGIN BAG - Final params: Start={params['med_start']}, Stop={final_stop}, Rate={rate:.1f}mL/h, Volume={volume:.1f}mL, Duration={duration:.2f}h")
                    
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
            
            # Get theoretical parameters for this rate change
            params = get_order_params_for_row(row, supertable)
                
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
            else:
                new_rate = original_rate
                new_volume = original_volume  
                new_duration = original_duration
                print("Rate change params failed validation - using original parameters")
            
            # Check if rate actually changed
            if pd.notna(new_rate) and pd.notna(prev_rate) and new_rate == prev_rate:
                print("Rate change but no actual rate difference - ignoring")
                continue
                
            # Check if rate change is valid
            if pd.isna(new_rate) or new_rate <= 0:
                print("Invalid new rate - ignoring rate change")
                continue
                
            # Check timing relative to previous infusion
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
                if actual_duration > 0:
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
                        continue
                
                # Add new period with remaining volume and new rate
                medsdict[order_id][med_name]["volume"].append(remaining_volume)
                medsdict[order_id][med_name]["rate"].append(new_rate)
                medsdict[order_id][med_name]["duration"].append(final_duration)
                medsdict[order_id][med_name]["med_start"].append(med_action_time)
                medsdict[order_id][med_name]["med_stop"].append(new_stop_time)
                
                print(f"RATE CHANGE (DURING INFUSION) - Final params: Start={med_action_time}, Stop={new_stop_time}, Rate={new_rate:.1f}mL/h, Volume={remaining_volume:.1f}mL, Duration={final_duration:.2f}h")
                
            else:
                # Rate change after previous infusion ended - start new infusion
                print(f"Rate change after infusion ended (ended at {prev_stop})")
                
                # Calculate actual duration for this new infusion period
                actual_duration, used_infuse_indices, last_infuse_time = calculate_actual_infusion_duration(
                    row, med_rows, processed_indices
                )
                processed_indices.update(used_infuse_indices)
                
                # Use provided volume or fall back to original
                if pd.notna(new_volume):
                    volume_to_use = new_volume
                    print(f"Using new volume: {volume_to_use:.1f}mL")
                elif pd.notna(original_volume):
                    volume_to_use = original_volume
                    print(f"Using original volume: {volume_to_use:.1f}mL")
                else:
                    volume_to_use = prev_volume
                    print(f"Using previous volume: {volume_to_use:.1f}mL")
                
                # Determine final duration
                if actual_duration > 0:
                    final_duration = actual_duration
                    new_stop_time = last_infuse_time
                    print(f"Using actual duration from Infuse actions: {final_duration:.2f}h")
                else:
                    # No infuse actions - use theoretical duration
                    if pd.notna(new_duration):
                        final_duration = new_duration
                    elif pd.notna(original_duration):
                        final_duration = original_duration
                    else:
                        final_duration = volume_to_use / new_rate if new_rate > 0 else 0
                    
                    new_stop_time = med_action_time + pd.Timedelta(hours=final_duration)
                    print(f"Using theoretical duration: {final_duration:.2f}h")
                
                # Add new infusion period
                medsdict[order_id][med_name]["volume"].append(volume_to_use)
                medsdict[order_id][med_name]["rate"].append(new_rate)
                medsdict[order_id][med_name]["duration"].append(final_duration)
                medsdict[order_id][med_name]["med_start"].append(med_action_time)
                medsdict[order_id][med_name]["med_stop"].append(new_stop_time)
                
                print(f"RATE CHANGE (AFTER INFUSION) - Final params: Start={med_action_time}, Stop={new_stop_time}, Rate={new_rate:.1f}mL/h, Volume={volume_to_use:.1f}mL, Duration={final_duration:.2f}h")
            
        elif med_action == "Infuse":
            # Check if this is the first action and no medication is set up yet
            if not medsdict[order_id][med_name]["set"] and row.name not in processed_indices:
                # Check if there's a "Begin Bag" within one hour after this Infuse
                upcoming_begin_bag = None
                for _, future_row in med_rows.iterrows():
                    if future_row.name <= row.name:
                        continue
                    future_time = pd.to_datetime(future_row["med_action_time"])
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
            
            # Check if med_stop is available (it should be for "Not Recorded" actions)
            if pd.isna(row["med_stop"]):
                print("Not Recorded action missing med_stop - this is unexpected, ignoring")
                continue
            
            med_start_time = pd.to_datetime(row["med_action_time"])
            med_stop_time = pd.to_datetime(row["med_stop"])
            duration = (med_stop_time - med_start_time).total_seconds() / 3600.0
            
            print(f"Not Recorded with med_start: {med_start_time}, med_stop: {med_stop_time}, duration: {duration:.2f}h")
            
            # Get order parameters for this row
            params = get_order_params_for_row(row, supertable)
            
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
                    # First infusion - establish parameters
                    medsdict[order_id][med_name]["volume"] = [volume]
                    medsdict[order_id][med_name]["rate"] = [rate]
                    medsdict[order_id][med_name]["duration"] = [duration]
                    medsdict[order_id][med_name]["med_start"] = [med_start_time]
                    medsdict[order_id][med_name]["med_stop"] = [med_stop_time]
                    medsdict[order_id][med_name]["set"] = True
                    
                    # Store original parameters for fallback
                    medsdict[order_id][med_name]["original_rate"] = rate
                    medsdict[order_id][med_name]["original_volume"] = volume
                    medsdict[order_id][med_name]["original_duration"] = duration
                    
                    print(f"NOT RECORDED (INITIAL) - Final params: Start={med_start_time}, Stop={med_stop_time}, Rate={rate:.1f}mL/h, Volume={volume:.1f}mL, Duration={duration:.2f}h")
                else:
                    # Subsequent infusion
                    medsdict[order_id][med_name]["volume"].append(volume)
                    medsdict[order_id][med_name]["rate"].append(rate)
                    medsdict[order_id][med_name]["duration"].append(duration)
                    medsdict[order_id][med_name]["med_start"].append(med_start_time)
                    medsdict[order_id][med_name]["med_stop"].append(med_stop_time)
                    
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
                    medsdict[order_id][med_name]["volume"].append(volume)
                    medsdict[order_id][med_name]["rate"].append(original_rate)
                    medsdict[order_id][med_name]["duration"].append(duration)
                    medsdict[order_id][med_name]["med_start"].append(med_start_time)
                    medsdict[order_id][med_name]["med_stop"].append(med_stop_time)
                    
                    print(f"NOT RECORDED (FALLBACK) - Final params: Start={med_start_time}, Stop={med_stop_time}, Rate={original_rate:.1f}mL/h, Volume={volume:.1f}mL, Duration={duration:.2f}h")
                else:
                    print("No existing parameters and invalid params for Not Recorded - ignoring")
                    
        elif med_action == "Bolus":
            processed_indices.add(row.name)
            print("Processing Bolus action")
            
            # Get order parameters for this bolus
            params = get_order_params_for_row(row, supertable)
            med_start_time = pd.to_datetime(row["med_action_time"])
            
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
                            print("No original parameters available for Bolus fallback - ignoring")
                            continue
                    else:
                        print("No volume available and no existing medication parameters - ignoring Bolus")
                        continue
                        
                med_stop_time = med_start_time + pd.Timedelta(hours=duration)
            
            # Add bolus to medsdict
            if not medsdict[order_id][med_name]["set"]:
                # First bolus - establish parameters
                medsdict[order_id][med_name]["volume"] = [volume]
                medsdict[order_id][med_name]["rate"] = [rate]
                medsdict[order_id][med_name]["duration"] = [duration]
                medsdict[order_id][med_name]["med_start"] = [med_start_time]
                medsdict[order_id][med_name]["med_stop"] = [med_stop_time]
                medsdict[order_id][med_name]["set"] = True
                
                # Store original parameters for fallback
                medsdict[order_id][med_name]["original_rate"] = rate
                medsdict[order_id][med_name]["original_volume"] = volume
                medsdict[order_id][med_name]["original_duration"] = duration
                
                print(f"BOLUS (INITIAL) - Final params: Start={med_start_time}, Stop={med_stop_time}, Rate={rate:.1f}mL/h, Volume={volume:.1f}mL, Duration={duration:.2f}h")
            else:
                # Subsequent bolus
                medsdict[order_id][med_name]["volume"].append(volume)
                medsdict[order_id][med_name]["rate"].append(rate)
                medsdict[order_id][med_name]["duration"].append(duration)
                medsdict[order_id][med_name]["med_start"].append(med_start_time)
                medsdict[order_id][med_name]["med_stop"].append(med_stop_time)
                
                print(f"BOLUS (SUBSEQUENT) - Final params: Start={med_start_time}, Stop={med_stop_time}, Rate={rate:.1f}mL/h, Volume={volume:.1f}mL, Duration={duration:.2f}h")
                
        else:
            # Handle other unrecorded med actions (not "Begin Bag", "Rate Change", "Infuse", "Not Recorded", or "Bolus")
            if not medsdict[order_id][med_name]["set"] and row.name not in processed_indices:
                print(f"Other unrecorded med action: {med_action} - checking if params are valid")
                params = get_order_params_for_row(row, supertable)
                
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


def process_encounter_new(meds, supertable):
    """
    NEW FUNCTION: Process encounter using improved infusion duration logic.
    """
    meds = meds.sort_values("med_action_time")
    imeds = meds.loc[meds["is_infusion"]]
    print(f"Initial infusion meds: {imeds.shape}")
    imeds = process_premix(imeds)
    imeds = imeds.loc[imeds.formulary_name != "Not Recorded"]
    print(f"After filtering: {imeds.shape}")
    
    unique_order_ids = imeds["order_med_id"].unique()
    print(f"{len(unique_order_ids)} unique order ids")
    medsdict = {}
    
    for order_id in unique_order_ids:
        print(f'\nProcessing order id: {order_id}')
        
        # Initialize medsdict for this order
        unique_meds = imeds.loc[imeds["order_med_id"] == order_id]['formulary_name'].unique()
        medsdict[order_id] = {}
        for med in unique_meds:
            medsdict[order_id][med] = {
                'rate': [], 'duration': [], 'med_start': [], 'med_stop': [], 'volume': [], 'set': False,
                'original_rate': None, 'original_volume': None, 'original_duration': None
            }
        
        # Get all rows for this order
        order_rows = imeds.loc[imeds["order_med_id"] == order_id].sort_values('med_action_time')
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


def process_encounter(meds, supertable):
    meds = meds.sort_values("med_action_time")
    imeds = meds.loc[meds["is_infusion"] ]
    print(imeds.shape)
    imeds = process_premix(imeds)
    imeds = imeds.loc[imeds.formulary_name != "Not Recorded"]
    print(imeds.shape)
    unique_order_ids = imeds["order_med_id"].unique()
    print(f"{len(unique_order_ids)} unique order ids")
    medsdict = {}

    for order_id in unique_order_ids:
        unique_meds = imeds.loc[imeds["order_med_id"] == order_id]['med_name'].unique()
        medsdict[order_id] = {}
        for med in unique_meds:
            medsdict[order_id][med] = {
                'rate' : [], 'duration' : [], 'med_start' : [], 'med_stop': [], 'volume': [], 'set' : False,
                'original_rate': None, 'original_volume': None, 'original_duration': None
            }

        rows = imeds.loc[imeds["order_med_id"] == order_id]
        print(f'Processing order id: {order_id}')
        # Group by formulary_name, med_action_time, and med_action
        for group_key, group_df in rows.groupby(['formulary_name', 'med_action_time', 'med_action']):
            formulary_name, med_action_time, med_action = group_key
            if len(group_df) > 1:        
                print("---------------------------------------------------------------")
            print(f"Processing group with {len(group_df)} rows: {formulary_name}, {med_action_time}, {med_action}")
            medsdict = process_rows(group_df, medsdict, supertable)
    
    return medsdict


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

