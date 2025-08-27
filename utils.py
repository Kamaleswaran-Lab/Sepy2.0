from typing import List
import yaml
import pandas as pd
import logging
import numpy as np
import re

def load_yaml(filename):
    """
    Load and parse a YAML file.
    Args:
        filename (str): The path to the YAML file to be loaded.
    Returns:
        dict: The contents of the YAML file as a dictionary.
    """
    with open(filename, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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


    
###########################################################################
############################# Summary Functions ###########################
###########################################################################
def sofa_summary(encounter_instance):
    """
    Summarizes the SOFA scores for a single patient encounter and returns a dataframe.

    Args:
        encounter_csn (str): The unique encounter ID (CSN) for the patient encounter.
        encounter_instance (sepyDICT): An instance of the sepyDICT class, containing the encounter data.
    """
    sofa_scores = (
        encounter_instance.clinical_data.sofa_scores
        .reset_index()
        .rename(columns={"index": "time_stamp"})
    )
    sofa_scores["csn"] = encounter_instance.clinical_data.csn
    return sofa_scores

def sepsis3_summary(encounter_instance):
    """
    Summarizes the Sepsis-3 time data for a single patient encounter and returns a dataframe.

    Args:
        encounter_csn (str): The unique encounter ID (CSN) for the patient encounter.
        encounter_instance (sepyDICT): An instance of the sepyDICT class, containing the encounter data.
    """
    sep3_time = encounter_instance.clinical_data.event_times['first_sep3_time_mod']
    sep3_time_df = pd.DataFrame(columns = ['csn', 'first_sep3_time_mod'], index = [0])
    sep3_time_df["csn"] = encounter_instance.clinical_data.csn
    sep3_time_df["first_sep3_time_mod"] = sep3_time
    return sep3_time_df

def sirs_summary(encounter_instance):
    """
    Summarizes the SIRS scores for a single patient encounter and returns a dataframe.

    Args:
        encounter_csn (str): The unique encounter ID (CSN) for the patient encounter.
        encounter_instance (sepyDICT): An instance of the sepyDICT class, containing the encounter data.
    """
    sirs_scores = (
        encounter_instance.clinical_data.sirs_scores
        .reset_index()
        .rename(columns={"index": "time_stamp"})
    )
    sirs_scores["csn"] = encounter_instance.clinical_data.csn
    return sirs_scores

def sepsis2_summary(encounter_instance):
    """
    Summarizes the Sepsis-2 time data for a single patient encounter and returns a dataframe.

    Args:
        encounter_csn (str): The unique encounter ID (CSN) for the patient encounter.
        encounter_instance (sepyDICT): An instance of the sepyDICT class, containing the encounter data.
    """
    sep2_time = encounter_instance.clinical_data.sep2_time
    sep2_time["csn"] = encounter_instance.clinical_data.csn
    return sep2_time

def enc_summary(encounter_instance):
    """
    Summarizes encounter-level data by combining flags, static features, and event times, then returns a dataframe.

    Args:
        csn (str): The unique encounter ID (CSN) for the patient encounter.
        encounter_instance (sepyDICT): An instance of the sepyDICT class, containing the encounter data, including flags, static features, and event times.
    """
    enc_summary_dict = {
        **encounter_instance.clinical_data.flags,
        **encounter_instance.clinical_data.static_features,
        **encounter_instance.clinical_data.event_times,
    }
    enc_summary_df = pd.DataFrame(enc_summary_dict, index=[0])
    enc_summary_df["csn"] = encounter_instance.clinical_data.csn
    enc_summary_df = enc_summary_df.set_index(["csn"])
    return enc_summary_df


def comorbidity_summary(encounter_instance, config_data):
    """
    Summarizes the comorbidity data for a single patient encounter based on a configuration file.

    Args:
        encounter_csn (str): The unique encounter ID (CSN) for the patient encounter.
        encounter_instance (sepyDICT): An instance of the sepyDICT class, containing comorbidity-related data.
        config_data (dict): A dictionary containing configuration data for comorbidity summaries.
    """
    encounter_csn = encounter_instance.clinical_data.csn
    comorbidity_summary_dicts = {}
    for summary_name in config_data['comorbidity_summary']:
        comorbidity_summary_dicts[summary_name + '_dict'] = {}
        
    for summary_name in config_data['comorbidity_summary']:
        try:
            comorbidity_summary_dicts[summary_name + '_dict'][encounter_csn] = getattr(encounter_instance.clinical_data, f"{summary_name}").icd_count
        except AttributeError:
            logging.warning(f"Attribute {summary_name} not found for csn {encounter_csn}")
        except KeyError as e:
            logging.error(f"Key error for {summary_name}_dict: {e}")
        except Exception as e:
            logging.error(f"Error processing comorbidity {summary_name} for csn {encounter_csn}: {e}")
    return comorbidity_summary_dicts

###########################################################################
############################## Data Cleaning ##############################
###########################################################################
def make_numeric(df, cols):
    """  
    Cleans and converts specified columns in a DataFrame to numeric format.  
    Args:  
        df (pandas.DataFrame): The DataFrame containing the columns to be processed.  
        cols (list): A list of column names to clean and convert to numeric values.  
    Returns:  
        pandas.DataFrame: The modified DataFrame with specified columns converted to numeric types.  
    """
    # Remove all the non-numeric characters from numerical cols
    df[cols] = df[cols].replace(r"\>|\<|\%|\/|\s", "", regex = True)
    # Converts specific cols to numeric
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    return df

###########################################################################
### Custom Date Parser to Handle Date Errors (i.e. coerce foolishness) ####
###########################################################################
def d_parser(s):
    """  
    Parses a given string or array-like object into a datetime format.  
    Args:  
        s (str, list, or pandas.Series): The input data to be converted to datetime.  
    Returns:  
        pandas.Series or pandas.DateTimeIndex: The parsed datetime object(s).  
    """
    return pd.to_datetime(s, errors = "coerce")


###########################################################################
############################# Read FLATFILES ###############################
###########################################################################

def read_data_file(file_path, index_col=None, date_cols=None, na_values=None, 
                   drop_cols=None, numeric_cols=None, low_memory=False, 
                   memory_map=False, date_parser=d_parser, header=0, dtype=None):
    """
    Generic function to read data files in various formats (CSV, DSV, pickle)
    with appropriate error handling and parameter support.
    
    Args:
        file_path (str): Path to the data file
        index_col (str or int, optional): Column to use as DataFrame index
        date_cols (list, optional): Columns to parse as datetime
        na_values (list, optional): Values to consider as NaN
        drop_cols (list, optional): Columns to drop from the DataFrame
        numeric_cols (list, optional): Columns to convert to numeric format
        low_memory (bool, optional): Whether to use memory-efficient parsing
        memory_map (bool, optional): Whether to memory-map the file
        date_parser (function, optional): Function to parse date strings
        header (int, optional): Row to use as column names
        dtype (dict, optional): Data types for specific columns
        
    Returns:
        pandas.DataFrame: The loaded data
        
    Raises:
        FileNotFoundError: If there's an error reading the file
    """
    logging.info(f"Reading file: {file_path}")
    
    try:
        # Determine file type and read accordingly
        if file_path.endswith(".csv"):
            logging.info(f"Reading CSV file: {file_path}")
            df = pd.read_csv(
                file_path,
                header=header,
                index_col=index_col,
                parse_dates=date_cols,
                na_values=na_values,
                low_memory=low_memory,
                memory_map=memory_map,
                date_parser=date_parser,
                dtype=dtype
            )
            
        elif file_path.endswith(".dsv"):
            logging.info(f"Reading DSV file: {file_path}")
            try:
                df = pd.read_csv(
                file_path,
                header=header,
                index_col=index_col,
                parse_dates=date_cols,
                na_values=na_values,
                sep="|", 
                low_memory=low_memory,
                memory_map=memory_map,
                date_parser=date_parser,
                dtype=dtype
                )
            except Exception as e:
                logging.error(f"Error reading file {file_path}: {str(e)}")
                logging.info(f"Attempting to read file with no pipe separator")    
                df = pd.read_csv(
                    file_path,
                    header=header,
                    index_col=index_col,
                    parse_dates=date_cols,
                    na_values=na_values,
                    low_memory=low_memory,  
                    memory_map=memory_map,
                    date_parser=date_parser,
                    dtype=dtype
                )
                logging.info(f"File read successfully with {df.shape[0]} rows and {df.shape[1]} columns")
            
        elif file_path.endswith(".pkl") or file_path.endswith(".pickle"):
            df = pd.read_pickle(file_path)
            
            # Apply date parsing if needed and file is a pickle
            if date_cols and isinstance(date_cols, list):
                for col in date_cols:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        
        else:
            # Default to CSV reading for unknown extensions
            logging.warning(f"Unknown file extension for {file_path}, attempting to read as CSV")
            try:
                df = pd.read_csv(
                file_path,
                header=header,
                index_col=index_col,
                parse_dates=date_cols,
                na_values=na_values,
                low_memory=low_memory,
                memory_map=memory_map,
                date_parser=date_parser,
                dtype=dtype
                )
            except FileNotFoundError as e:
                logging.error(f"Error reading file {file_path}: {str(e)}")
                raise FileNotFoundError(f"Error reading file {file_path}: {str(e)}")
            
        # Post-processing: drop columns if specified
        if drop_cols:
            df = df.drop(columns=drop_cols)
            
        # Convert numeric columns if specified
        if numeric_cols:
            # Remove non-numeric characters and convert to numeric
            df[numeric_cols] = df[numeric_cols].replace(r"\>|\<|\%|\/|\s", "", regex=True)
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
            
        logging.info(f"Successfully read file with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
        
    except FileNotFoundError as e:
        error_msg = f"Error reading file {file_path}: {str(e)}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

def read_flatfile(file_path):
    """
    Reads a flatfile and returns a pandas DataFrame.
    """
    return pd.read_csv(file_path)


###########################################################################
##################### Aggregate Utility Functions #########################
###########################################################################
def get_bounds(var_name, bounds):
    df = bounds.loc[bounds['location in supertable'] == var_name]
    upperbound = df['physical upper bound'].values[0]
    lowerbound = df['physical lower bound'].values[0]
    
    # Convert strings or invalid entries to np.nan
    try:
        upperbound = float(upperbound)
    except (ValueError, TypeError):
        upperbound = np.nan
    try:
        lowerbound = float(lowerbound)
    except (ValueError, TypeError):
        lowerbound = np.nan

    return lowerbound, upperbound


class BoundAggregator:
    def __init__(self, lowerbound, upperbound, operation='mean'):
        self.lowerbound = lowerbound
        self.upperbound = upperbound
        self.operation = operation
    
    def __call__(self, array):
        try:
            array = array.astype(float)
        except (TypeError, ValueError):
            return np.nan
        
        if np.isnan(array).all():
            return np.nan
        
        values = array[~np.isnan(array)]
        if not np.isnan(self.lowerbound):
            values = values[values >= self.lowerbound]
        if not np.isnan(self.upperbound):
            values = values[values <= self.upperbound]

        if len(values) == 0:
            return np.nan
            
        if self.operation == 'mean':
            return np.mean(values)
        elif self.operation == 'min':
            return np.min(values)
        elif self.operation == 'max':
            return np.max(values)

def agg_fn_wrapper(var_name, bounds):
    try:
        lowerbound, upperbound = get_bounds(var_name, bounds)
    except Exception as e:
        logging.error(f"No bounds found for {var_name}")
        return "mean"
    
    return BoundAggregator(lowerbound, upperbound, 'mean')

def agg_fn_wrapper_min(var_name, bounds):
    lowerbound, upperbound = get_bounds(var_name, bounds)
    return BoundAggregator(lowerbound, upperbound, 'min')

def agg_fn_wrapper_max(var_name, bounds):
    lowerbound, upperbound = get_bounds(var_name, bounds)
    return BoundAggregator(lowerbound, upperbound, 'max')