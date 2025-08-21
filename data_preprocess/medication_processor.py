import pandas as pd
import numpy as np
import sys
sys.path.append("../")
import utils


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
    elif (pd.isna(rate_parent) and pd.isna(volume_parent) and 
          abs(duration_parent - (volume_parent / rate_parent)) > 10):
        suspicious = True 
    
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
    premix["checked"] = [
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
    is_fluid = row["is_fluids"]

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
    weight = supertable['daily_weight_kg'].values[supertable.index.get_loc(pd.to_datetime(row["med_action_time"]), method = 'nearest')]
    parent_order_params = extract_parent_order_params(row)
    clinical_desc_params = extract_clinical_desc_params(row)
    reconciled_params = reconcile_parameters(parent_order_params, clinical_desc_params)
    infusion_params = extract_infusion_params(row, weight)
    med_start = pd.to_datetime(row["med_action_time"])

    volume = reconciled_params["volume"]
    rate = reconciled_params["rate"]
    duration = reconciled_params["duration"]
    final_check = reconciled_params["final_check"]
    print("Reconciled params:") 
    print(f"Volume: {volume}, Rate: {rate}, Duration: {duration}, Final Check: {final_check}")
     
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
            if infusion_params["amount_inf_unit"].lower() in infusion_params["rate_unit"].lower():
                duration = infusion_params["amount_inf"] / infusion_params["rate_inf"]
                rate = infusion_params["volume_inf"] / duration
                final_check = True
                if (pd.notna(infusion_params["med_stop"])) & (duration != infusion_params["duration_inf"]):
                    final_check = False
                else:
                    final_check = True
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
        print("Using reconciled params")
        final_check = True
        
    if (pd.notna(infusion_params["med_stop"])):
        med_stop = infusion_params["med_stop"] 
    elif pd.notna(duration) and duration != 0.0: 
        med_stop = med_start + pd.Timedelta(hours=duration)
    else:
        med_stop = np.nan 
        
    return {
        "volume": volume,
        "rate": rate,
        "duration": duration,
        "final_check": final_check,
        "med_start": med_start,
        "med_stop": med_stop
    }

def process_medication_row(row, meds, supertable, ongoing_infusion, meds_dict):
    """
    Process a single medication row to extract infusion parameters.
    
    Args:
        row: Single row from medications DataFrame
        meds: Complete medications DataFrame
        supertable: Patient data table (for weight lookup)
        ongoing_infusion: Dict tracking ongoing infusions
        meds_dict: Dict counting medications by formulary name
        
    Returns:
        dict: Processed medication data with extracted parameters
    """
    # Skip if this is a continuation of an ongoing infusion
    if (row["order_med_id"] in ongoing_infusion and 
        row["med_action"] == "Infuse"):
        return None
    
    # Track medication counts
    formulary_name = row["formulary_name"]
    if formulary_name not in meds_dict:
        meds_dict[formulary_name] = 1
    else:
        meds_dict[formulary_name] += 1
    
    # Get patient weight
    try:
        weight = supertable.loc[row["med_action_time"]].iloc[0]
    except (KeyError, IndexError):
        weight = row.get("weight", 70.0)  # Default weight if not found
    
    # Extract parameters from different sources
    parent_data = extract_parent_order_params(row)
    clinical_data = extract_clinical_desc_params(row)
    
    # Reconcile parameters
    reconciled = reconcile_parameters(parent_data, clinical_data)
    
    volume = reconciled["volume"]
    rate = reconciled["rate"]
    duration = reconciled["duration"]
    
    # Try to extract volume from formulary name if not found
    if not volume:
        volume_info = utils.extract_volume_detailed(formulary_name)
        if volume_info["unit"] == "mL":
            volume = volume_info["raw_value"]
    
    # Calculate dose-based rates if available
    dose_data = calculate_dose_based_rate(row, weight)
    if dose_data["rate"] is not None:
        rate = dose_data["rate"]
    if dose_data["duration"] is not None:
        duration = dose_data["duration"]
    
    # Extract start/stop times if available
    med_start = None
    med_stop = None
    if not pd.isna(row.get("med_start")):
        med_start = pd.to_datetime(row["med_start"])
        if not pd.isna(row.get("med_stop")):
            med_stop = pd.to_datetime(row["med_stop"])
    
    # Calculate duration for Begin Bag actions
    if row["med_action"] == "Begin Bag":
        calculated_duration = calculate_infusion_duration(row, meds, ongoing_infusion)
        if calculated_duration is not None:
            duration = calculated_duration
    
    # Calculate rate from volume and duration if start/stop times available
    if med_start and med_stop and volume:
        duration_from_times = (med_stop - med_start).total_seconds() / 3600.0
        if duration_from_times == 0.0:
            rate = volume  # Instantaneous administration
        else:
            rate = volume / duration_from_times
            duration = duration_from_times
    elif med_start and duration and reconciled["final_check"]:
        med_stop = med_start + pd.Timedelta(hours=duration)
    
    
    return {
        "order_med_id": row["order_med_id"],
        "formulary_name": formulary_name,
        "med_action_time": row["med_action_time"],
        "med_action": row.get("med_action"),
        "volume": volume,
        "rate": rate,
        "duration": duration,
        "med_start": med_start,
        "med_stop": med_stop,
        "weight": weight,
        "source": reconciled["source"],
        "final_check": reconciled["final_check"]
    }


def process_medication_infusions(meds, supertable, supertable_index):
    """
    Process all medication records and return structured results.
    
    Args:
        meds: DataFrame with medication records
        supertable: Patient data for weight lookup
        supertable_index: Time index for supertable
        
    Returns:
        tuple: (processed_df, medication_counts)
    """
    # Initialize tracking dictionaries
    ongoing_infusion = {}
    meds_dict = {}
    
    processed_records = []
    
    # Process each medication row
    for index, row in meds.iterrows():
        result = process_medication_row(row, meds, supertable, ongoing_infusion, meds_dict)
        if result is not None:
            processed_records.append(result)
    
    # Convert to DataFrame
    if processed_records:
        processed_df = pd.DataFrame(processed_records)
        processed_df['med_action_time'] = pd.to_datetime(processed_df['med_action_time'])
        processed_df = processed_df.set_index('med_action_time')
    else:
        processed_df = pd.DataFrame()
    
    return processed_df, meds_dict