from pathlib import Path
import pandas as pd
import numpy as np
import sys
sys.path.append("../")
import utils
import tqdm

import re
from typing import Optional, Dict, List

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
    
    def normalize_unit(unit):
        """Remove trailing 's' from unit names for matching."""
        if unit.endswith('s'):
            return unit[:-1]
        return unit
    
    # Get conversion factors with normalization
    from_factor = unit_conversion_factors.get(normalize_unit(from_unit))
    to_factor = unit_conversion_factors.get(normalize_unit(to_unit))
    
    if from_factor is None or to_factor is None:
        return None  # Cannot convert
        
    # Convert: amount * (from_factor / to_factor)
    return amount * (from_factor / to_factor)

def impute_by_closest_location_vectorized(df):
    """
    Impute 'Not Recorded' values with the closest non-'Not Recorded' value
    within the same order_med_id and med_name group
    """
    df = df.copy()
    
    # Store imputation values to apply all at once
    imputation_dict = {}
    
    # Group by order_med_id and med_name
    for (order_id, med_name), group in tqdm.tqdm(df.groupby(['order_med_id', 'med_name'])):
        # Split into recorded and not recorded
        not_recorded_mask = group['formulary_name'] == 'Not Recorded'
        not_recorded_indices = group.index[not_recorded_mask].values
        recorded_indices = group.index[~not_recorded_mask].values
        
        if len(recorded_indices) == 0 or len(not_recorded_indices) == 0:
            continue
        
        # Vectorized distance calculation using broadcasting
        # Shape: (len(not_recorded), len(recorded))
        distances = np.abs(not_recorded_indices[:, None] - recorded_indices[None, :])
        
        # Find closest recorded entry for each not recorded entry
        closest_recorded_positions = np.argmin(distances, axis=1)
        closest_recorded_indices = recorded_indices[closest_recorded_positions]
        
        # Get the values from the closest recorded entries
        closest_values = df.loc[closest_recorded_indices, 'formulary_name'].values
        
        # Store imputation mappings
        for not_rec_idx, value in zip(not_recorded_indices, closest_values):
            imputation_dict[not_rec_idx] = value
    
    # Apply all imputations at once (much faster than individual .loc assignments)
    if imputation_dict:
        df.loc[list(imputation_dict.keys()), 'formulary_name'] = list(imputation_dict.values())
    
    return df

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

###### PARSE AND APPLY THE DFAULT CONCENTRATIONS OR RATES #############################

def parse_concentration_default(conc_str: Optional[str]) -> Dict[str, Optional[float]]:
    """
    Extract amount, amount_unit, and volume from concentration strings.
    
    Args:
        conc_str: String like "500 mg / 250 mL" or "200 mcg / mL" or "?" or None
    
    Returns:
        Dictionary with keys: 'amount', 'amount_unit', 'volume_ml'
        For rate concentrations (e.g., "200 mcg / mL"), returns amount per 1 mL
        Returns None values if parsing fails or input is missing/invalid
    """
    # Handle None, empty, or "?" inputs
    if not conc_str or conc_str.strip() == "?":
        return {'amount': None, 'amount_unit': None, 'volume_ml': None}
    
    # Pattern 1: <number> <unit> / <number> mL (e.g., "500 mg / 250 mL")
    pattern_full = r'([\d.]+)\s*(mg|mcg|gm|g|units?|mEq)\s*/\s*([\d.]+)\s*mL'
    match = re.search(pattern_full, conc_str.strip(), re.IGNORECASE)
    
    if match:
        amount = float(match.group(1))
        amount_unit = match.group(2)
        volume_ml = float(match.group(3))
        
        # Normalize unit names
        if amount_unit.lower() == 'gm':
            amount_unit = 'g'
        
        return {
            'amount': amount,
            'amount_unit': amount_unit,
            'volume_ml': volume_ml
        }
    
    # Pattern 2: <number> <unit> / mL (e.g., "200 mcg / mL") - concentration rate
    pattern_rate = r'([\d.]+)\s*(mg|mcg|gm|g|units?|mEq)\s*/\s*mL'
    match = re.search(pattern_rate, conc_str.strip(), re.IGNORECASE)
    
    if match:
        amount = float(match.group(1))
        amount_unit = match.group(2)
        volume_ml = 1.0  # Per 1 mL
        
        # Normalize unit names
        if amount_unit.lower() == 'gm':
            amount_unit = 'g'
        
        return {
            'amount': amount,
            'amount_unit': amount_unit,
            'volume_ml': volume_ml
        }
    
    # If it doesn't match either pattern (like "0.9 mg / kg")
    return {'amount': None, 'amount_unit': None, 'volume_ml': None}

# Calculate actual volume based on formulary name amount
def calculate_volume_from_concentration(row):
    """
    Calculate volume based on formulary amount and parsed concentration.
    
    If formulary has amount_inf:
        - Convert amount_inf to milligrams
        - Use concentration (mg/mL) to calculate volume
    Else:
        - Use concentration_default volume directly
    """
    # Check if we have a valid concentration
    if pd.isna(row['conc_mg_per_ml']) and pd.isna(row['conc_volume_ml']):
        return None
    
    # If formulary name has an amount, use it with the concentration
    if pd.notna(row['amount_inf']) and pd.notna(row['amount_inf_unit']):
        # Map amount_inf_unit to standardized unit
        amount_unit_std = amount_unit_mapping.get(
            row['amount_inf_unit'].lower(),
            row['amount_inf_unit']
        )
        
        # Convert formulary amount to milligrams
        amount_inf_mg = convert_units(
            row['amount_inf'],
            amount_unit_std,
            'Milligram'
        )
        
        if amount_inf_mg is not None and pd.notna(row['conc_mg_per_ml']) and row['conc_mg_per_ml'] > 0:
            # Calculate volume from amount and concentration
            return amount_inf_mg / row['conc_mg_per_ml']
    
    # Otherwise, use the scaled volume from concentration_default
    return row['conc_volume_ml']


def parse_rate_default(rate_str):
    """
    Extract numeric rate value from rate_default strings (always in ml/hr).
    
    Args:
        rate_str: String like "50 ml / hr", "25 mL/hour", "100ml/hr", etc.
    
    Returns:
        float: Numeric rate in ml/hr, or None if parsing fails
    """
    if pd.isna(rate_str) or not rate_str:
        return None
    
    # Pattern to match: <number> followed by ml/hr variations
    # Handles: "50 ml / hr", "25 mL/hour", "100ml/hr", etc.
    pattern = r'([\d.]+)\s*ml\s*/?\s*h'
    
    match = re.search(pattern, str(rate_str).strip(), re.IGNORECASE)
    
    if match:
        return float(match.group(1))
    
    return None



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


##########################################################3
############# Fluids Parsing Functions ####################
##########################################################3


def extract_numbers(text_list: List[str], match_string: str = None):
        """
        Extract and return numbers from a list of text strings.
        If match_string is provided, find numbers that appear before the match string (separated by space).
        
        Args:
            text_list: List of strings potentially containing numbers
            match_string: Optional string to match; if provided, only numbers before this string are returned
            
        Returns:
            List of numeric values found, or None if no valid numbers
        """
        if not text_list:
            return None
            
        try:
            # Remove commas and extract numbers
            cleaned_texts = [text.replace(',', '') for text in text_list]
            all_numbers = []
            
            for text in cleaned_texts:
                if match_string:
                    # Pattern to find number followed by space and then the match string
                    pattern = re.compile(rf"([-+]?(?:\d*\.\d+|\d+))\s+{re.escape(match_string)}")
                    matches = pattern.findall(text)
                    all_numbers.extend([float(num) for num in matches])
                else:
                    # extract all numbers
                    NUMBER_PATTERN = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")
                    numbers = NUMBER_PATTERN.findall(text)
                    all_numbers.extend([float(num) for num in numbers])
            
            return all_numbers if all_numbers else None
                
        except (ValueError, TypeError) as e:
            logging.warning(f"Error extracting numbers from {text_list}: {e}")
            
        return None
    
def parse_clinical_description(description: str):
    """
    Parse clinical description to extract 1. volume 2. rate 3. duration 4. bolus 5. Amount (if applicable)
    Args:
        description: Clinical description text
    Returns:
        dict: A dictionary containing the extracted parameters
    """
    BOLUS_PATTERN = re.compile(r'\bbolus\b', re.IGNORECASE)
    VOLUME_PATTERN1 = re.compile(r'\bTotal Volume\b', re.IGNORECASE)
    VOLUME_PATTERN2 = re.compile(r'\bmL\b(?!/hr)', re.IGNORECASE)
    RATE_PATTERN = re.compile(r'\bmL/hr\b', re.IGNORECASE)
    TIME_HR_PATTERN = re.compile(r'\bhr\(s\)?\b', re.IGNORECASE)
    TIME_MIN_PATTERN = re.compile(r'\bminute\(s\)?\b', re.IGNORECASE)

    AMOUNT_PATTERN = re.compile(r'\b(?:mEq|meq|gm|g|mg|mcg|unit|units|meq/kg/min|mEq/kg/min|gm/kg/min|g/kg/min|mg/kg/min|mcg/kg/min|gm/kg/hr|g/kg/hr|mg/kg/hr|mcg/kg/hr|unit/kg/min|unit/kg/hr|units/kg/min|units/kg/hr|meq/kg/hr|mEq/kg/hr)\b', re.IGNORECASE)

    # Split description into components using ", " and "; "
    desc_parts = [part.strip() for part in re.split(r', |; ', description)]
    
    # Initialize parameters
    params = {"volume": None, "volume_unit": None, "rate": None, "rate_unit": None, 
              "duration": None, "duration_unit": None, "bolus": None, 
              "amount": None, "amount_unit": None}

    # Extract bolus
    bolus_parts = [part for part in desc_parts if BOLUS_PATTERN.search(part)]
    if bolus_parts:
        params['bolus'] = extract_numbers(bolus_parts)
    
    # Extract amount
    amount_parts = [part for part in desc_parts if AMOUNT_PATTERN.search(part)]
    if amount_parts:
        # Extract the actual matched unit strings from each part
        amount_units = []
        amounts = []
        for part in amount_parts:
            matches = AMOUNT_PATTERN.findall(part)
            for match in matches:
                number = extract_numbers([part], match)
                if number:
                    amount_units.append(match)
                    amounts.append(number[0])
        params['amount_unit'] = amount_units if amount_units else None
        params['amount'] = amounts if amounts else None
          

    # Extract volume (mL but not mL/hr)
    volume_parts = []
    volume_units = []
    volumes = []
    for part in desc_parts:
        if VOLUME_PATTERN1.search(part):
            volume_parts.append(part)
            volume_units.append('total_volume')
            volume = extract_numbers([part])
            if volume:
                volumes.append(volume[0])
            else:
                volumes.append(None)
        elif VOLUME_PATTERN2.search(part):
            volume_parts.append(part)
            matches = VOLUME_PATTERN2.findall(part)
            for match in matches:
                number = extract_numbers([part], match)
                if number:
                    volumes.append(number[0])
                    volume_units.append('mL')

    params['volume_unit'] = volume_units if volume_units else None
    params['volume'] = volumes if volumes else None

    # Extract rate (mL/hr)
    rate_parts = [part for part in desc_parts if RATE_PATTERN.search(part)]
    if rate_parts:
        # Handle case where 'mL/hr' appears alone
        processed_rate_parts = []
        for i, part in enumerate(desc_parts):
            if part.strip() == 'mL/hr' and i > 0:
                processed_rate_parts.append(desc_parts[i-1])
            elif RATE_PATTERN.search(part) and part.strip() != 'mL/hr':
                processed_rate_parts.append(part)
        
        if processed_rate_parts:
            params['rate'] = extract_numbers(processed_rate_parts)
            params['rate_unit'] = ['mL/hr'] 
    
    # Extract time duration
    time_parts = []
    time_units = []
    for part in desc_parts:
        if TIME_HR_PATTERN.search(part):
            time_parts.append(part)
            time_units.append('hr')
        elif TIME_MIN_PATTERN.search(part):
            time_parts.append(part)
            time_units.append('min')
    
    if time_parts:
        duration_value = extract_numbers(time_parts)
        
        if duration_value:
            # Convert minutes to hours, keep hours as is
            converted_duration = []
            for i in range(len(duration_value)):
                if time_units[i] == 'min':
                    converted_duration.append(duration_value[i]/60.0)
                else:
                    converted_duration.append(duration_value[i])
            duration_value = converted_duration
            params['duration'] = duration_value
            params['duration_unit'] = ['hr']*len(duration_value)
    
    # Calculate missing parameters
    #try:
    #    if params['volume'] and params['rate'] and not params['duration']:
    #        params['duration'] = params['volume'] / params['rate']
    #    elif params['volume'] and params['duration'] and not params['rate']:
    #        params['rate'] = params['volume'] / params['duration']
    #    elif params['rate'] and params['duration'] and not params['volume']:
    #        params['volume'] = params['rate'] * params['duration']
    #except (ZeroDivisionError, TypeError) as e:
    #    logging.warning(f"Error calculating parameters: {e}")  
            
    return params
    
       
def extract_volume_detailed(formulary_name):
    """
    Extract volume with more detailed parsing, including multiple volume mentions.
    
    Args:
        formulary_name (str): The medication name to extract volume from
        
    Returns:
        dict: Dictionary with 'volume', 'unit', 'raw_value', and 'all_volumes'
    """
    if not formulary_name or not isinstance(formulary_name, str):
        return {
            'volume': 'none',
            'unit': None,
            'raw_value': None,
            'all_volumes': []
        }
    
    name = formulary_name.lower()
    all_volumes = []
    
    # All patterns with named groups for better extraction
    patterns = [
        r'(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ml)\b',
        r'/(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ml)\b',
        r'(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>l)\b(?!\w)',
        r'/(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>l)\b(?!\w)',
        r'(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>cc)\b',
        r'/(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>cc)\b',
        r'(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>liter[s]?)\b',
        r'(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>gallon[s]?)\b',
        r'(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>fl\s*oz)\b',
        r'(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>pint[s]?)\b',
        r'(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>quart[s]?)\b',
    ]
    
    # Find all volume mentions
    for pattern in patterns:
        matches = re.finditer(pattern, name, re.IGNORECASE)
        for match in matches:
            value = match.group('value')
            unit_raw = match.group('unit').lower()
            
            # Standardize unit names
            if unit_raw == 'ml':
                unit = 'mL'
            elif unit_raw == 'l':
                unit = 'L'
            elif unit_raw == 'cc':
                unit = 'cc'
            elif 'liter' in unit_raw:
                unit = 'L'
            elif 'gallon' in unit_raw:
                unit = 'gal'
            elif 'fl' in unit_raw and 'oz' in unit_raw:
                unit = 'fl oz'
            elif 'pint' in unit_raw:
                unit = 'pt'
            elif 'quart' in unit_raw:
                unit = 'qt'
            else:
                unit = unit_raw
            
            volume_str = f"{value} {unit}"
            all_volumes.append({
                'value': float(value),
                'unit': unit,
                'formatted': volume_str
            })
    
    if not all_volumes:
        return {
            'volume': 'none',
            'unit': None,
            'raw_value': None,
            'all_volumes': []
        }
    
    # Return the first (primary) volume found
    primary = all_volumes[0]
    return {
        'volume': primary['formatted'],
        'unit': primary['unit'],
        'raw_value': primary['value'],
        'all_volumes': all_volumes
    }