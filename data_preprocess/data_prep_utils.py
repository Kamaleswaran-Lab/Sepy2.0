import pandas as pd
from pathlib import Path
import re

def merge_o2_flow_and_vent_data(df_o2_flow: pd.DataFrame, 
                                df_vent: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the o2 flow and vent data s expected by SepyImport.

    Args:
        df_o2_flow: The o2 flow data.
        df_vent: The vent data.
        o2_result_col: The column name of the o2 result.
        o2_units_col: The column name of the o2 units.
        o2_recorded_time_col: The column name of the o2 recorded time.
        o2_csn_col: The column name of the o2 csn.
        o2_patient_id_col: The column name of the o2 patient id.

    Returns:
        The merged dataframe.
    """
    df_o2_flow['recorded_time'] = pd.to_datetime(df_o2_flow['recorded_time'])
    df_vent['recorded_time'] = pd.to_datetime(df_vent['recorded_time'])
    
    df_vent = df_vent.drop(columns=['o2_flow_rate'])
    df_o2_flow['result_tval'] = pd.to_numeric(df_o2_flow['result_tval'], errors='coerce')
    df_o2_flow = df_o2_flow.rename(columns={'encounter_nbr': 'csn', 'result_tval': 'oxygen_flow_rate', 'unit_measure': 'oxygen_flow_rate_units'})
    df_o2_flow = df_o2_flow[['recorded_time', 'csn', 'oxygen_flow_rate', 'oxygen_flow_rate_units']]
    merged_df = pd.merge(
                    df_vent, 
                    df_o2_flow, 
                    on=['csn', 'recorded_time'], 
                    how='outer'
                )
    return merged_df

#Function to convert icd9 column to icd10 codes in a dataframe according to the mapping in the icd10toicd9gem file 
def convert_icd9_to_icd10(df: pd.DataFrame, icd9_col: str, icd10_col: str, mapping_file: str) -> pd.DataFrame:
    """
    Convert icd9 column to icd10 codes in a dataframe.
    """
    #Read the mapping file
    mapping_df = pd.read_csv(mapping_file)

    #Create a dictionary of the mapping
    mapping = dict(zip(mapping_df["icd9cm"], mapping_df["icd10cm"]))

    #Apply the mapping to the dataframe
    df[icd10_col] = df[icd9_col].map(mapping)
    
    return df

def safe_read_dsv(file_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, sep="|")
        if df.shape[1] == 1:
            df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    

#### CODE TO EXTRACT VOLUME FROM INFUSION MEDS ####

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
    volume_patterns = [
        # mL variations
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>ml)\b',
        r'/(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>ml)\b',
        
        # Liter variations (with negative lookbehind/lookahead to avoid matching "LR", "chloride", etc.)
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>l)\b(?!\w)',
        r'/(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>l)\b(?!\w)',
        
        # cc (cubic centimeter) variations
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>cc)\b',
        r'/(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>cc)\b',
        
        # Other volume units
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>liter[s]?)\b',
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>gallon[s]?)\b',
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>fl\s*oz)\b',
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>pint[s]?)\b',
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>quart[s]?)\b',
    ]
    
    # Find all volume mentions
    for pattern in volume_patterns:
        matches = re.finditer(pattern, name, re.IGNORECASE)
        for match in matches:
            value_str = match.group('value')
            value = float(value_str.replace(',', ''))
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
                'value': value,
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

def extract_amount_detailed(formulary_name):
    """
    Extract medication amount(s) with units from a formulary name.
    
    Args:
        formulary_name (str): The medication name to extract amount from
        
    Returns:
        dict: Dictionary with 'amount', 'unit', 'raw_value', and 'all_amounts'
    """
    if not formulary_name or not isinstance(formulary_name, str):
        return {
            'amount': 'none',
            'unit': None,
            'raw_value': None,
            'all_amounts': []
        }
    
    name = formulary_name.lower()
    all_amounts = []
    
    # Comprehensive patterns for medication amounts with named groups
    amount_patterns = [
        # Weight-based units
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>mg)\b',          # milligrams
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>g)\b(?!al)',     # grams (but not "gal")
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>gm)\b',          # grams (gm variant)
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>kg)\b',          # kilograms
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>mcg)\b',         # micrograms
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>μg)\b',          # micrograms (mu symbol)
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>ng)\b',          # nanograms
        
        # Activity units
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>unit[s]?)\b',    # units
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>u)\b(?!\w)',     # units (single letter)
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>iu)\b',          # international units
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>miu)\b',         # million international units
        
        # Equivalents and molarity
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>meq)\b',         # milliequivalents
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>eq)\b',          # equivalents
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>mmol)\b',        # millimoles
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>mol)\b',         # moles
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>osmol)\b',       # osmoles
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>mosm)\b',        # milliosmoles
        
        # Percentage and concentration
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>%)\b',           # percentage
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>ppm)\b',         # parts per million
        
        # Special pharmaceutical units
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>drops?)\b',      # drops
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>gtt)\b',         # drops (abbreviation)
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>sprays?)\b',     # sprays
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>puffs?)\b',      # puffs
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>tabs?)\b',       # tablets
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>caps?)\b',       # capsules
        
        # Radioactivity units
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>mci)\b',         # millicuries
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>ci)\b',          # curies
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>mbq)\b',         # megabecquerels
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>bq)\b',          # becquerels
        
        # Time-related units (for sustained release)
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>hr)\b',          # hours
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>hours?)\b',      # hours
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>min)\b',         # minutes
        r'(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>minutes?)\b',    # minutes
    ]
    
    # Find all amount mentions
    for pattern in amount_patterns:
        matches = re.finditer(pattern, name, re.IGNORECASE)
        for match in matches:
            value_str = match.group('value')
            value = float(value_str.replace(',', ''))
            unit_raw = match.group('unit').lower()
            
            # Standardize unit names
            unit = standardize_unit(unit_raw)
            
            amount_str = f"{value} {unit}"
            all_amounts.append({
                'value': value,
                'unit': unit,
                'formatted': amount_str,
                'raw_unit': unit_raw
            })
    
    if not all_amounts:
        return {
            'amount': 'none',
            'unit': None,
            'raw_value': None,
            'all_amounts': []
        }
    
    # Return the first (primary) amount found
    primary = all_amounts[0]
    return {
        'amount': primary['formatted'],
        'unit': primary['unit'],
        'raw_value': primary['value'],
        'all_amounts': all_amounts
    }

def standardize_unit(unit_raw):
    """
    Standardize unit names to consistent format.
    
    Args:
        unit_raw (str): Raw unit string from regex match
        
    Returns:
        str: Standardized unit name
    """
    unit_map = {
        # Weight units
        'mg': 'mg',
        'g': 'g',
        'kg': 'kg',
        'mcg': 'mcg',
        'μg': 'mcg',
        'ng': 'ng',
        
        # Activity units
        'unit': 'unit',
        'units': 'unit',
        'u': 'unit',
        'iu': 'IU',
        'miu': 'MIU',
        
        # Equivalents
        'meq': 'mEq',
        'eq': 'Eq',
        'mmol': 'mmol',
        'mol': 'mol',
        'osmol': 'osmol',
        'mosm': 'mOsm',
        
        # Concentration
        '%': '%',
        'ppm': 'ppm',
        
        # Special units
        'drop': 'drop',
        'drops': 'drop',
        'gtt': 'gtt',
        'spray': 'spray',
        'sprays': 'spray',
        'puff': 'puff',
        'puffs': 'puff',
        'tab': 'tab',
        'tabs': 'tab',
        'cap': 'cap',
        'caps': 'cap',
        
        # Radioactivity
        'mci': 'mCi',
        'ci': 'Ci',
        'mbq': 'MBq',
        'bq': 'Bq',
        
        # Time
        'hr': 'hr',
        'hour': 'hr',
        'hours': 'hr',
        'min': 'min',
        'minute': 'min',
        'minutes': 'min',
    }
    
    return unit_map.get(unit_raw, unit_raw)

def extract_concentration(formulary_name):
    """
    Extract concentration information (amount per volume) from formulary names.
    
    Args:
        formulary_name (str): The medication name to extract concentration from
        
    Returns:
        dict: Dictionary with concentration information
    """
    if not formulary_name or not isinstance(formulary_name, str):
        return {
            'concentration': 'none',
            'amount': None,
            'amount_unit': None,
            'volume': None,
            'volume_unit': None
        }
    
    name = formulary_name.lower()
    
    # Patterns for concentration (amount/volume)
    concentration_patterns = [
        r'(?P<amount>\d+(?:\.\d+)?)\s*(?P<amount_unit>mg|g|mcg|μg|ng|meq|mmol|unit[s]?|u|iu)\s*/\s*(?P<volume>\d+(?:\.\d+)?)\s*(?P<volume_unit>ml|l|cc)',
        r'(?P<amount>\d+(?:\.\d+)?)\s*(?P<amount_unit>mg|g|mcg|μg|ng|meq|mmol|unit[s]?|u|iu)\s*per\s*(?P<volume>\d+(?:\.\d+)?)\s*(?P<volume_unit>ml|l|cc)',
    ]
    
    for pattern in concentration_patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            amount_val = float(match.group('amount'))
            amount_unit = standardize_unit(match.group('amount_unit').lower())
            volume_val = float(match.group('volume'))
            volume_unit = match.group('volume_unit').upper()
            
            concentration_str = f"{amount_val} {amount_unit}/{volume_val} {volume_unit}"
            
            return {
                'concentration': concentration_str,
                'amount': amount_val,
                'amount_unit': amount_unit,
                'volume': volume_val,
                'volume_unit': volume_unit
            }
    
    return {
        'concentration': 'none',
        'amount': None,
        'amount_unit': None,
        'volume': None,
        'volume_unit': None
    }

def process_medication_csv_with_amounts(input_dataframe = None, input_file = None, output_file=None):
    """
    Process the medication CSV file to extract amounts, concentrations, and volumes.
    
    Args:
        input_dataframe (pd.DataFrame): Input dataframe with medication data
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file (optional)
        
    Returns:
        pd.DataFrame: Processed DataFrame with extracted information
    """
    # Read the CSV file
    if isinstance(input_dataframe, pd.DataFrame):
        df = input_dataframe
    else:
        try:
            df = pd.read_csv(input_file)
            print(f"Successfully loaded CSV with {len(df)} rows")
        except FileNotFoundError:
            print(f"Error: File '{input_file}' not found")
            return None
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return None
    
    # Add new columns for extraction
    df['primary_amount'] = None
    df['amount_unit'] = None
    df['amount_numeric'] = None
    df['concentration'] = None
    df['volume_given'] = None
    df['volume_unit'] = None
    df['volume_numeric'] = None
    df["is_infusion"] = True   
    df["is_anesthesia"] = False
    
    # Process each row
    amount_count = 0
    concentration_count = 0
    volume_count = 0
    
    for idx, row in df.iterrows():
        formulary_name = row['formulary_name']
        
        # Extract amounts
        amount_info = extract_amount_detailed(formulary_name)
        df.at[idx, 'primary_amount'] = amount_info['amount']
        df.at[idx, 'amount_unit'] = amount_info['unit']
        df.at[idx, 'amount_numeric'] = amount_info['raw_value']
        
        # Extract concentrations
        conc_info = extract_concentration(formulary_name)
        df.at[idx, 'concentration'] = conc_info['concentration']
        
        # Extract volumes (from previous function)
        volume_info = extract_volume_detailed(formulary_name)
        df.at[idx, 'volume_given'] = volume_info['volume']
        df.at[idx, 'volume_unit'] = volume_info['unit']
        df.at[idx, 'volume_numeric'] = volume_info['raw_value']
        
        # Count extractions
        if amount_info['amount'] != 'none':
            amount_count += 1
        if conc_info['concentration'] != 'none':
            concentration_count += 1
        if volume_info['volume'] != 'none':
            volume_count += 1
        
        # check if formulary name has indicators for it not being an infusion
        terms_to_exclude = ["inj", "syringe", "tab", "cap", "inhale",
                            "inhalation", "inhaler", "supp", "oral", "aero",
                            "epidural", "granules", "spray", "gel", "patch",
                            "syr", "lozenge"] 
        if any(term in formulary_name.lower() for term in terms_to_exclude):
            df.at[idx, 'is_infusion'] = False
        else: 
            df.at[idx, 'is_infusion'] = True
        
        if "ANES" in formulary_name:
            df.at[idx, 'is_anesthesia'] = True
        else:
            df.at[idx, 'is_anesthesia'] = False
        
    # Print statistics
    print(f"\nProcessing complete!")
    print(f"Entries with amounts extracted: {amount_count}")
    print(f"Entries with concentrations extracted: {concentration_count}")
    print(f"Entries with volumes extracted: {volume_count}")
    print(f"Total entries processed: {len(df)}")
    
    # Show unit distributions
    print(f"\nAmount unit distribution:")
    amount_unit_counts = df[df['amount_unit'].notna()]['amount_unit'].value_counts()
    print(amount_unit_counts.head(10))
    
    print(f"\nVolume unit distribution:")
    volume_unit_counts = df[df['volume_unit'].notna()]['volume_unit'].value_counts()
    print(volume_unit_counts)
    
    # Save to output file if specified
    if output_file:
        try:
            df.to_csv(output_file, index=False)
            print(f"\nProcessed file saved as: {output_file}")
        except Exception as e:
            print(f"Error saving file: {e}")
    
    return df

def process_medication_csv_with_extraction(input_file, output_file=None):
    """
    Process the medication CSV file to extract actual volume values.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file (optional)
        
    Returns:
        pd.DataFrame: Processed DataFrame with extracted volumes
    """
    # Read the CSV file
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully loaded CSV with {len(df)} rows")
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        return None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
    
    # Add new columns for volume extraction
    df['volume_given'] = None
    df['volume_unit'] = None
    df['volume_numeric'] = None
    
    # Process each row
    volume_count = 0
    none_count = 0
    
    for idx, row in df.iterrows():
        formulary_name = row['formulary_name']
        
        # Extract volume using detailed function
        volume_info = extract_volume_detailed(formulary_name)
        
        df.at[idx, 'volume_given'] = volume_info['volume']
        df.at[idx, 'volume_unit'] = volume_info['unit']
        df.at[idx, 'volume_numeric'] = volume_info['raw_value']
        
        if volume_info['volume'] != 'none':
            volume_count += 1
        else:
            none_count += 1
    
    # Print statistics
    print(f"\nProcessing complete!")
    print(f"Entries with volume extracted: {volume_count}")
    print(f"Entries with no volume: {none_count}")
    print(f"Total entries processed: {len(df)}")
    
    # Show volume distribution
    print(f"\nVolume unit distribution:")
    unit_counts = df[df['volume_unit'].notna()]['volume_unit'].value_counts()
    print(unit_counts)
    
    # Show some examples
    print(f"\nExamples of extracted volumes:")
    volume_examples = df[df['volume_given'] != 'none'][['formulary_name', 'volume_given']].head(10)
    for idx, row in volume_examples.iterrows():
        print(f"'{row['formulary_name']}' -> {row['volume_given']}")
    
    # Save to output file if specified
    if output_file:
        try:
            df.to_csv(output_file, index=False)
            print(f"\nProcessed file saved as: {output_file}")
        except Exception as e:
            print(f"Error saving file: {e}")
    
    return df

def main():
    #root = Path("/hpc/group/kamaleswaranlab/EmoryDataset/EMR_RAW/noPHI")

    #for year in range(2015, 2022):
    #    vent_file = root / f"{year}" / f"CJSEPSIS_VITALS_{year}.dsv"
    #    o2_file = root / f"{year}" / f"vent_o2_flow_rate{year}.dsv"
        
    #    df_vent = safe_read_dsv(vent_file)
    #    df_o2_flow = safe_read_dsv(o2_file)
        
    #    merged_df = merge_o2_flow_and_vent_data(df_o2_flow, df_vent)
        
    #    merged_df.to_csv(root / f"{year}" / f"VITALS_O2_FLOW_RATE_{year}.csv", index=False)
    #    print(f"Processed {year}")
    infusion_meds_mapping = pd.read_csv("../groupings/em_infusion_meds_volume.csv")
    df = process_medication_csv_with_amounts(input_dataframe = infusion_meds_mapping)
    """
    df["is_fluids"] = False 
    fluid_cols = [
		'Sodium Chloride 0.9% intravenous solution',
		'Lactated Ringers Injection intravenous solution',
		'Sodium Chloride 0.45% intravenous solution',
		'Dextrose 5% with 0.2% NaCl and KCl 20 mEq/L intravenous solution',
		'potassium chloride-sodium chloride',
		'Dextrose 5% in Lactated Ringers intravenous solution',
		'Dextrose 20% in Water intravenous solution',
		'Dextrose 5% in Water with KCl 20 mEq/l intravenous solution',
		'Dextrose 5% in Lactated Ringers with KCl 20 mEq/l intravenous solution',
		'dextran, low molecular weight',
		'sodium chloride, hypertonic, ophthalmic',
		'Electrolyte (Plasma-Lyte) intravenous solution',
		'Albumin 5%', 'albumin human', 'albumin 25%']
    fluid_cols = [col.lower() for col in fluid_cols]
    df.loc[df["med_name_generic"].str.lower().isin(fluid_cols), "is_fluids"] = True """
    df.to_csv("../groupings/em_infusion_meds_volume_amounts.csv", index=False)

if __name__ == "__main__":
    main()
        
        