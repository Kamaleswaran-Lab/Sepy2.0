import yaml
import pandas as pd
import logging
import numpy as np

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
    
    
###########################################################################
############################# Summary Functions ###########################
###########################################################################
def sofa_summary(encounter_csn, encounter_instance):
    """
    Summarizes the SOFA scores for a single patient encounter and returns a dataframe.

    Args:
        encounter_csn (str): The unique encounter ID (CSN) for the patient encounter.
        encounter_instance (sepyDICT): An instance of the sepyDICT class, containing the encounter data.
    """
    sofa_scores = (
        encounter_instance.encounter_dict["sofa_scores"]
        .reset_index()
        .rename(columns={"index": "time_stamp"})
    )
    sofa_scores["csn"] = encounter_csn
    return sofa_scores

def sepsis3_summary(encounter_csn, encounter_instance):
    """
    Summarizes the Sepsis-3 time data for a single patient encounter and returns a dataframe.

    Args:
        encounter_csn (str): The unique encounter ID (CSN) for the patient encounter.
        encounter_instance (sepyDICT): An instance of the sepyDICT class, containing the encounter data.
    """
    sep3_time = encounter_instance.encounter_dict["sep3_time"]
    sep3_time["csn"] = encounter_csn
    return sep3_time

def sirs_summary(encounter_csn, encounter_instance):
    """
    Summarizes the SIRS scores for a single patient encounter and returns a dataframe.

    Args:
        encounter_csn (str): The unique encounter ID (CSN) for the patient encounter.
        encounter_instance (sepyDICT): An instance of the sepyDICT class, containing the encounter data.
    """
    sirs_scores = (
        encounter_instance.encounter_dict["sirs_scores"]
        .reset_index()
        .rename(columns={"index": "time_stamp"})
    )
    sirs_scores["csn"] = encounter_csn
    return sirs_scores

def sepsis2_summary(encounter_csn, encounter_instance):
    """
    Summarizes the Sepsis-2 time data for a single patient encounter and returns a dataframe.

    Args:
        encounter_csn (str): The unique encounter ID (CSN) for the patient encounter.
        encounter_instance (sepyDICT): An instance of the sepyDICT class, containing the encounter data.
    """
    sep2_time = encounter_instance.encounter_dict["sep2_time"]
    sep2_time["csn"] = encounter_csn
    return sep2_time

def enc_summary(encounter_instance):
    """
    Summarizes encounter-level data by combining flags, static features, and event times, then returns a dataframe.

    Args:
        csn (str): The unique encounter ID (CSN) for the patient encounter.
        encounter_instance (sepyDICT): An instance of the sepyDICT class, containing the encounter data, including flags, static features, and event times.
    """
    enc_summary_dict = {
        **encounter_instance.flags,
        **encounter_instance.static_features,
        **encounter_instance.event_times,
    }
    enc_summary_df = pd.DataFrame(enc_summary_dict, index=[0]).set_index(["csn"])
    return enc_summary_df

def create_comorbidity_summary_dicts(config_data):
    """
    Creates a dictionary of comorbidity summary dictionaries based on the configuration data.

    Args:
        config_data (dict): A dictionary containing configuration data for comorbidity summaries.
    """
    comorbidity_summary_dicts = {}
    for summary_name in config_data['comorbidity_summary']:
        comorbidity_summary_dicts[summary_name + '_dict'] = {}
    return comorbidity_summary_dicts

def comorbidity_summary(encounter_csn, encounter_instance, config_data, comorbidity_summary_dicts):
    """
    Summarizes the comorbidity data for a single patient encounter based on a configuration file.

    Args:
        encounter_csn (str): The unique encounter ID (CSN) for the patient encounter.
        encounter_instance (sepyDICT): An instance of the sepyDICT class, containing comorbidity-related data.
        config_data (dict): A dictionary containing configuration data for comorbidity summaries.
        comorbidity_summary_dicts (dict): A dictionary of comorbidity summary dictionaries.
    """
    for summary_name in config_data['comorbidity_summary']:
        try:
            comorbidity_summary_dicts[summary_name + '_dict'][encounter_csn] = getattr(encounter_instance, f"{summary_name}_PerCSN").icd_count
        except AttributeError:
            logging.warning(f"Attribute {summary_name}_PerCSN not found for csn {encounter_csn}")
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
        ValueError: If there's an error reading the file
    """
    logging.info(f"Reading file: {file_path}")
    
    try:
        # Determine file type and read accordingly
        if file_path.endswith(".csv"):
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
        
    except Exception as e:
        error_msg = f"Error reading file {file_path}: {str(e)}"
        logging.error(error_msg)
        raise ValueError(error_msg)

def read_flatfile(file_path):
    """
    Reads a flatfile and returns a pandas DataFrame.
    """
    return pd.read_csv(file_path)


###########################################################################
##################### Aggregate Utility Functions #########################
###########################################################################
def get_bounds(var_name, bounds):
    df = bounds.loc[bounds['Location in SuperTable'] == var_name]
    upperbound = df['Physical Upper bound'].values[0]
    lowerbound = df['Physical lower bound'].values[0]
    
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


def agg_fn_wrapper(var_name, bounds):
    lowerbound, upperbound = get_bounds(var_name, bounds)

    def agg_fn(array):
        try:
            array = array.astype(float)
        except (TypeError, ValueError):
            return np.nan
        
        if np.isnan(array).all():
            return np.nan
        
        values = array[~np.isnan(array)]
        if not np.isnan(lowerbound):
            values = values[values >= lowerbound]
        if not np.isnan(upperbound):
            values = values[values <= upperbound]

        return np.mean(values) if len(values) > 0 else np.nan

    return agg_fn


def agg_fn_wrapper_min(var_name, bounds):
    lowerbound, upperbound = get_bounds(var_name, bounds)

    def agg_fn(array):
        try:
            array = array.astype(float)
        except (TypeError, ValueError):
            return np.nan
        
        if np.isnan(array).all():
            return np.nan
        
        values = array[~np.isnan(array)]
        if not np.isnan(lowerbound):
            values = values[values >= lowerbound]
        if not np.isnan(upperbound):
            values = values[values <= upperbound]

        return np.min(values) if len(values) > 0 else np.nan

    return agg_fn


def agg_fn_wrapper_max(var_name, bounds):
    lowerbound, upperbound = get_bounds(var_name, bounds)

    def agg_fn(array):
        try:
            array = array.astype(float)
        except (TypeError, ValueError):
            return np.nan
        
        if np.isnan(array).all():
            return np.nan
        
        values = array[~np.isnan(array)]
        if not np.isnan(lowerbound):
            values = values[values >= lowerbound]
        if not np.isnan(upperbound):
            values = values[values <= upperbound]

        return np.max(values) if len(values) > 0 else np.nan

    return agg_fn