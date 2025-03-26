# -*- coding: utf-8 -*-
"""
This module provides functions to import data from flat files into pandas dataframes.
The dataframes are then pickled for later use in super table construction.

Elite Data Hacks
Author: Christopher S. Josef, MD
Email: csjosef@krvmail.com
Version: 0.1

Kameleswaran Labs
Author: Jack F. Regan
Edited: 2025-03-01
Version: 0.2
Changes:
     - update dictionary paths to by dynamically generated.
     - update documentation.
     - added configuration file through yaml.
"""
import pickle
import time
import glob
import sys
import yaml

import sepyIMPORT as si

############################## Load YAML ##############################
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
yaml_data = load_yaml(str(sys.argv[2]))
############################## File Paths ##############################
# Generate file paths
### data path is the parent directory for all the flat files; you'll specify each file location below
# CSJPC data_path = "C:/Users/DataSci/Desktop/em_data"
# OD data_path = "C:/Users/DataSci/OneDrive - Emory University/Sepsis Calculation/Data_Export"
# CLUSTER
DATA_PATH = yaml_data["data_path"]
### grouping path is where the lists of meds, labs, & comorbs will be located
# CSJ PC groupings_path = "C:/Users/DataSci/Documents/GitHub/sepy/0.grouping"
# OD groupings_path = "C:/Users/DataSci/OneDrive - Emory University/Sepsis Calculation/groupings"
# CLUSTER
GROUPINGS_PATH = yaml_data["groupings_path"]
### Output paths is where the pickles will be written
# CSJ PC output_path = "XXXX"
# OD output_path = "C:/Users/DataSci/OneDrive - Emory University/CJ_Sepsis/5.quality_check/"
# CLUSTER
OUTPUT_PATH = yaml_data["output_path"]
############################## File Dictionaries ##############################
def generate_paths(data_year):
    """
    Generates a dictionary of file paths for comorbidities, emergency medicine data,
    and year-based data files.
    Parameters:
       year (int): The year for which the data paths should be generated.
    Returns:
       paths (dict): A dictionary mapping descriptive keys to file paths.
    """
    paths = {}
    # load path types from yaml
    comorbidity_types = yaml_data["dictionary_paths"]["comorbidity_types"]
    grouping_types = yaml_data["dictionary_paths"]["grouping_types"]
    year_types = yaml_data["dictionary_paths"]["year_types"]
    for comorbidity in comorbidity_types:
        paths[f"path_comorbid_{comorbidity}"] = glob.glob(
            f"{GROUPINGS_PATH}/comorbidities/{comorbidity}.csv"
        )[0]
    for type in grouping_types:
        paths[f"path_{type}"] = glob.glob(f"{GROUPINGS_PATH}/{type}*.csv")[0]
    for year_type in year_types:
        paths[f"path_{year_type}"] = glob.glob(
            f"{DATA_PATH}/{data_year}/*_{year_type}*.dsv"
        )[0]
    return paths
############################## Import Data Frames ##############################
def import_data_frames(yearly_instance):
    """
    Imports data from a YAML structure and applies it to methods of a given instance.
    Args:
        yearly_instance (sepyIMPORT): The instance whose methods will be called.
    """
    import_start_time = time.time()
    print(
        "Sepy is currently reading flat files and importing them for analysis. Thank you for waiting."
    )
    for method_name, params in yaml_data["yearly_instance"].items():
        method = getattr(yearly_instance, method_name, None)
        if callable(method):
            if "numeric_cols" in params and isinstance(params["numeric_cols"], str):
                params["numeric_cols"] = getattr(yearly_instance, params["numeric_cols"], None)
            method(**params)
    print(f"Sepy took {time.time() - import_start_time} (s) to create a yearly pickle.")
############################## Main Function ##############################
if __name__ == "__main__":
    # Usage:
    #   python make_pickle.py <year> <CONFIGURATION_PATH>
    # Parameters:
    #   <year> (int): The year for which data is being processed.
    # Error Handling:
    #    - FileNotFoundError: Raised when the file or path dictionary cannot be found.
    #    - ValueError: Raised when the provided `year` argument is invalid (non-integer or out of range).
    #    - KeyError: Raised if the dynamically generated path dictionary name does not correspond to any valid dictionary.
    try:
        # Generate paths for each year dynamically and store them in a dictionary
        path_dictionary = {}
        for year in range(2014, 2022):
            path_dictionary[year] = generate_paths(year)
        # starts yearly pickle timer
        start = time.perf_counter()
        # accepts command line arguments
        year = int(sys.argv[1])
        # creates pickle file name
        PICKLE_FILE_NAME = OUTPUT_PATH + "em_y" + str(year) + ".pickle"
        print(PICKLE_FILE_NAME)
        # creates path dictionary name
        PATH_DICTIONARY = "path_dictionary" + str(year)
        print(f"File locations were taken from the path dictionary: {PATH_DICTIONARY}")
        import_instance = si.sepyIMPORT(path_dictionary[year], "|")
        print(f"An instance of the sepyIMPORT class was created for {year}")
        # import data frames from the sepyIMPORT instance and pickle data
        import_data_frames(import_instance)
        with open(PICKLE_FILE_NAME, "wb") as handle:
            pickle.dump(import_instance, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(
            f"Time to create {year}s data and write to pickles was {time.perf_counter()-start} (s)"
        )
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e)
        print(f"There was an error with the class instantiation for {year}")