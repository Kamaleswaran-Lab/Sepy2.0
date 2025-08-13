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
    root = Path("/hpc/group/kamaleswaranlab/EmoryDataset/EMR_RAW/noPHI")

    for year in range(2015, 2022):
        vent_file = root / f"{year}" / f"CJSEPSIS_VITALS_{year}.dsv"
        o2_file = root / f"{year}" / f"vent_o2_flow_rate{year}.dsv"
        
        df_vent = safe_read_dsv(vent_file)
        df_o2_flow = safe_read_dsv(o2_file)
        
        merged_df = merge_o2_flow_and_vent_data(df_o2_flow, df_vent)
        
        merged_df.to_csv(root / f"{year}" / f"VITALS_O2_FLOW_RATE_{year}.csv", index=False)
        print(f"Processed {year}")
        
if __name__ == "__main__":
    main()
        
        