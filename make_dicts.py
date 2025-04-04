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
import logging

import sepyIMPORT as si
import sepyDICT as sd
import pandas as pd
import numpy as np

from pathlib import Path
d
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
PICKLE_OUTPUT_PATH = yaml_data["pickle_output_path"]
### bed unit csv is a mapping of bed units to icu type [ed, ward, icu]
BED_UNIT_CSV_FNAME = Path(yaml_data["ben_unit_cs_fname"])
###
VARIABLE_BOUNDS_CSV_FNAME = Path(yaml_data["variable_bounds_csv_fname"])
###
DIALYSIS_INFO_CSN_FNAME = Path(yaml_data["dialysis_info_csn_fname"])
###
DICTIONARY_OUTPUT_PATH = yaml_data["dictionary_output_path"]
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
    logging.info(
        "Sepy is currently reading flat files and importing them for analysis. Thank you for waiting."
    )
    for method_name, params in yaml_data["yearly_instance"].items():
        method = getattr(yearly_instance, method_name, None)
        if callable(method):
            if "numeric_cols" in params and isinstance(params["numeric_cols"], str):
                params["numeric_cols"] = getattr(yearly_instance, params["numeric_cols"], None)
            method(**params)
    logging.info(f"Sepy took {time.time() - import_start_time} (s) to create a yearly pickle.")
############################## Make Dictionaries #############################
def process_csn(
    encounter_csn,
    pickle_save_path,
    bed_unit_mapping,
    thresholds,
    dialysis_info,
    yearly_data_instance,
):
    """
    Processes a single patient encounter (CSN) and serializes the encounter data to a pickle file.

    Args:
        encounter_csn (int or str): The unique encounter ID (CSN) to process.
        pickle_save_path (Path): The directory path where the pickle file will be saved.
        bed_unit_mapping (dict): A mapping of bed locations to ICU units.
        thresholds (dict): A dictionary containing threshold values or limits used in processing.
        dialysis_info (dict): Information related to dialysis treatment for the patient.
        yearly_data_instance (object): An instance of the `sepyDICT` class containing the yearly data.
    Returns:
        sepyDICT: An instance of the `sepyDICT` class containing the processed encounter data.
    """
    file_name = pickle_save_path / (str(encounter_csn) + ".pickle")
    # instantiate class for single encounter
    encounter_instance = sd.sepyDICT(
        yearly_data_instance, encounter_csn, bed_unit_mapping, thresholds, dialysis_info
    )
    # create encounter dictionary
    dictionary_instance = encounter_instance.encounter_dict
    # create a pickle file for encounter
    picklefile = open(file_name, "wb")
    # pickle the encounter dictionary and write it to file
    pickle.dump(dictionary_instance, picklefile)
    # close the file
    picklefile.close()
    # return dictionary for summary report functions
    return encounter_instance
############################ Summary Functions ##########################
def sofa_summary(encounter_csn, encounter_instance):
    """
    Summarizes the SOFA scores for a single patient encounter and appends the data to the global list.

    Args:
        encounter_csn (int or str): The unique encounter ID (CSN) for the patient encounter.
        encounter_instance (sepyDICT): An instance of the sepyDICT class, containing the encounter data.
    """
    sofa_scores = (
        encounter_instance.encounter_dict["sofa_scores"]
        .reset_index()
        .rename(columns={"index": "time_stamp"})
    )
    sofa_scores["csn"] = encounter_csn # add csn to sofa_scores
    appended_sofa_scores.append(sofa_scores)


def sepsis3_summary(encounter_csn, encounter_instance):
    """
    Summarizes the Sepsis-3 time data for a single patient encounter and appends it to the global list.

    Args:
        encounter_csn (int or str): The unique encounter ID (CSN) for the patient encounter.
        encounter_instance (sepyDICT): An instance of the sepyDICT class, containing the encounter data.
    """
    sep3_time = encounter_instance.encounter_dict["sep3_time"]
    sep3_time["csn"] = encounter_csn  # add csn to sep3 time
    appended_sep3_time.append(sep3_time)


def sirs_summary(encounter_csn, encounter_instance):
    """
    Summarizes the SIRS scores for a single patient encounter and appends the data to the global list.

    Args:
        encounter_csn (int or str): The unique encounter ID (CSN) for the patient encounter.
        encounter_instance (sepyDICT): An instance of the sepyDICT class, containing the encounter data.
    """
    sirs_scores = (
        encounter_instance.encounter_dict["sirs_scores"]
        .reset_index()
        .rename(columns={"index": "time_stamp"})
    )
    sirs_scores["csn"] = encounter_csn  # add csn to sirs_scores
    appended_sirs_scores.append(sirs_scores)


def sepsis2_summary(encounter_csn, encounter_instance):
    """
    Summarizes the Sepsis-2 time data for a single patient encounter and appends it to the global list.

    Args:
        encounter_csn (int or str): The unique encounter ID (CSN) for the patient encounter.
        encounter_instance (sepyDICT): An instance of the sepyDICT class, containing the encounter data.
    """
    sep2_time = encounter_instance.encounter_dict["sep2_time"]
    sep2_time["csn"] = encounter_csn  # add csn to sep3 time
    appended_sep2_time.append(sep2_time)


def enc_summary(encounter_instance):
    """
    Summarizes encounter-level data by combining flags, static features, and event times, then appends it to the global list.

    Args:
        csn (int or str): The unique encounter ID (CSN) for the patient encounter.
        encounter_instance (sepyDICT): An instance of the sepyDICT class, containing the encounter data, including flags, static features, and event times.
    """
    enc_summary_dict = {
        **encounter_instance.flags,
        **encounter_instance.static_features,
        **encounter_instance.event_times,
    }
    enc_summary_df = pd.DataFrame(enc_summary_dict, index=[0]).set_index(["csn"])
    appended_enc_summaries.append(enc_summary_df)

def comorbidity_summary(encounter_csn, encounter_instance):
    """
    Summarizes the comorbidity data for a single patient encounter based on a configuration file.

    Args:
        encounter_csn (int or str): The unique encounter ID (CSN) for the patient encounter.
        encounter_instance (sepyDICT): An instance of the sepyDICT class, containing comorbidity-related data.
    """
    for summary_name, summary_config in yaml_data['comorbidity_summaries'].items():
        if summary_config['enabled']:
            try:
                comorbidity_summary_dicts[summary_name + '_dict'][encounter_csn] = getattr(encounter_instance, f"{summary_name}_PerCSN").icd_count
            except AttributeError:
                logging.warning(f"Attribute {summary_name}_PerCSN not found for csn {encounter_csn}")
            except KeyError as e:
                logging.error(f"Key error for {summary_name}_dict: {e}")
            except Exception as e:
                logging.error(f"Error processing comorbidity {summary_name} for csn {encounter_csn}: {e}")

    logging.info(f'The csn for comorbids is: {encounter_csn}')

############################## Initialize Empty Summaries ##############################
yaml_data = yaml.safe_load(open('config.yaml')) # load the yaml file
comorbidity_summary_dicts = {}

for summary_name, summary_config in yaml_data['comorbidity_summaries'].items():
    comorbidity_summary_dicts[summary_name + '_dict'] = {}
# other summaries
appended_sofa_scores = []
appended_sep3_time = []
appended_sirs_scores = []
appended_sep2_time = []
appended_enc_summaries = []

start = time.perf_counter()
############################## Main Function ##############################
if __name__ == "__main__":
    # Usage:
    #   python make_dicts.py <year> <CONFIGURATION_PATH>
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
        logging.info(PICKLE_FILE_NAME)
        # creates path dictionary name
        PATH_DICTIONARY = "path_dictionary" + str(year)
        logging.info(f"File locations were taken from the path dictionary: {PATH_DICTIONARY}")
        import_instance = si.sepyIMPORT(path_dictionary[year], "|")
        logging.info(f"An instance of the sepyIMPORT class was created for {year}")
        # import data frames from the sepyIMPORT instance and pickle data
        import_data_frames(import_instance)
        with open(PICKLE_FILE_NAME, "wb") as handle:
            pickle.dump(import_instance, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(
            f"Time to create {year}s data and write to pickles was {time.perf_counter()-start} (s)"
        )
    except (FileNotFoundError, ValueError, KeyError) as e:
        logging.error(e)
        logging.error(f"There was an error with the class instantiation for {year}")
    ######################### Begin Dictionary Construction #############################
    # The script processes patient encounter data for a specific year, generating summaries on sepsis,
    # comorbidities, and encounter details while handling errors during processing. It writes the
    # results, including encounter summaries and error logs, to designated output directories.
    # accepts the arguments from the command line
    # Usage: python em_make_dicts.py <csn_list_file> <pickle_directory> <output_directory> <num_processes> <task_number> <year>

    # Expected User Input:
    #  csn_list_file (str): Path to the file containing the CSN list.
    #  pickle_directory (str): Path to the directory containing yearly pickles.
    #  output_directory (str): Path to the directory where output files will be stored.
    #  num_processes (int): Total number of processes to be used.
    #  task_number (int): The specific process assignment for this task.
    #  year (int): The year of the data being processed.
    try:
        logging.info(sys.version)
        csn_list_file_name = Path(sys.argv[1])
        # imports file location for CSN list
        logging.info(f"MkDct- The CSN List location: {csn_list_file_name}") # WHAT IS THIS FILE
        # sets directoy for yearly pickles
        pickle_path = PICKLE_OUTPUT_PATH
        logging.info(f"MkDct- The pickle directory: {PICKLE_OUTPUT_PATH}")
        # sets directory for output
        output_path = Path(sys.argv[3])
        logging.info(f"MkDct- The output directory: {DICTIONARY_OUTPUT_PATH}")
        # total number of tasks/processes
        num_processes = int(sys.argv[4]) # WHAT DOES THIS REFER TO?
        logging.info(f"MkDct- The tot num of processes: {num_processes}")
        # task number
        processor_assignment = int(sys.argv[5]) # WHAT DOES THIS REFER TO?
        logging.info(f"MkDct- This task is: {processor_assignment}")
        # year num provided in bash
        logging.info(f"MkDct- The import year is: {year}")
    except (IndexError, ValueError, TypeError) as e:
        logging.error(
            f"MkDct- There was an error importing one of the arguments: {type(e).__name__}."
        )
        logging.info(f"MkDct- You are trying to load the following CSN list {year}")
        logging.info(f"MkDct- You are trying to use this many processors {sys.argv[5]}") ## UPDATE TO REFLECT

    ########### Create Encounter List Based on Processor Assignment ###########
    # Cohort selector
    ed = 0
    in_pt = 1
    icu = 0
    adult = 1
    vent_row = 0
    vent_start = 0
    # reads the list of csns
    csn_df = pd.read_csv(csn_list_file_name, sep = "|", header = 0)
    # only keep csns that meet specified year
    csn_df = csn_df[(csn_df.in_pt == in_pt) & (csn_df.adult == adult)]
    logging.info(f"MkDct- A total of {csn_df.shape[0]} encounters were selected")
    csn_df = csn_df[["csn", "year"]]
    # drop duplicates and filter out only selected years
    csn_df = csn_df.drop_duplicates()
    csn_df = csn_df[csn_df.year == year]
    total_num_enc = len(csn_df)
    logging.info(f"MkDct- The year {year} has {total_num_enc} encounters.")
    # breaks encounter list into chunks, selects correct chunk based on process num
    chunk_size = int(total_num_enc / num_processes)
    logging.info(f"MkDct- The ~chunk size is {chunk_size}")
    # split list
    list_of_chunks = np.array_split(csn_df, num_processes)
    logging.info(f"MkDct- The list of chunks has {len(list_of_chunks)} unique dataframes.")
    # uses processor assignment to select correct chunk
    process_list = list_of_chunks[processor_assignment]["csn"]
    logging.info(f"MkDct- The process_list head:\n {process_list.head()}")
    # select correct pickle by year
    pickle_name = pickle_path / ("em_y" + str(year) + ".pickle")
    logging.info(f"MkDct- The following pickle is being read: {pickle_name}")
    try:
        # reads the IMPORT class instance (i.e.  1 year of patient data)
        pickle_load_time = time.perf_counter()
        yearly_instance = pd.read_pickle(pickle_name)
        logging.info(
            f"MkDct-Pickle from year {year} was loaded in {time.perf_counter()-pickle_load_time}s."
        )
        logging.info("-----------LOADED YEARLY PICKLE FILE!!!!---------------")
        # if success, make a dir for this year's encounters
        pickle_write_path = output_path / str(year)
        pickle_write_path.mkdir(exist_ok = True)
        logging.info(f"MkDct-Directory for year {year} was set to {pickle_write_path}")
        # make empty list to handle csn's with errors
        error_list = []
        ######################### Load Files for Extra Processing #########################
        start_csn_creation = time.perf_counter()
        bed_to_unit_mapping = pd.read_csv(BED_UNIT_CSV_FNAME)
        bed_to_unit_mapping.drop(columns=["Unnamed: 0"], inplace=True)
        try:
            bed_to_unit_mapping.columns = [
                "bed_unit",
                "icu_type",
                "unit_type",
                "hospital",
            ]
        except ValueError:
            logging.error("MkDct- The bed to unit mapping file is not formatted correctly.")
        bounds = pd.read_excel(VARIABLE_BOUNDS_CSV_FNAME, engine="openpyxl")
        dialysis = pd.read_csv(DIALYSIS_INFO_CSN_FNAME)
        dialysis_year = dialysis.loc[
            dialysis["Encounter Number"].isin(csn_df["csn"].values)
        ]
        ################################ Make Dicts by CSN ################################
        for count, csn in enumerate(process_list, start=1):
            try:
                logging.info(f"MkDct- Processing patient csn: {csn}, {count} of {chunk_size} for year {year}")
                instance = process_csn(csn, pickle_write_path, bed_to_unit_mapping, bounds, dialysis_year, yearly_instance)
                logging.info("MkDct- Instance created")
                # Running summaries with error handling
                try:
                    sofa_summary(csn, instance)
                except Exception as e:
                    logging.error(f"MkDct- Error in Sofa Summary for csn {csn}: {e}")
                try:
                    sepsis3_summary(csn, instance)
                except Exception as e:
                    logging.error(f"MkDct- Error in Sepsis 3 Summary for csn {csn}: {e}")
                try:
                    sirs_summary(csn, instance)
                except Exception as e:
                    logging.error(f"MkDct- Error in SIRS Summary for csn {csn}: {e}")
                try:
                    sepsis2_summary(csn, instance)
                except Exception as e:
                    logging.error(f"MkDct- Error in Sepsis 2 Summary for csn {csn}: {e}")
                try:
                    enc_summary(instance)
                except Exception as e:
                    logging.error(f"MkDct- Error in Encounter Summary for csn {csn}: {e}")
                try:
                    comorbidity_summary(csn, instance)
                except Exception as e:
                    logging.error(f"MkDct- Error in Comorbidity Summary for csn {csn}: {e}")
                logging.info(f"MkDct- Encounter {count} of {chunk_size} is complete!")
            except Exception as e:
                logging.error(f"MkDct- Error processing csn {csn}: {e}")
                error_list.append([csn, e.args[0]])
                logging.error(f"MkDct- The following csn had an error: {csn}")
        ############################# Export Sepsis Summary #############################
        # create sepsis_summary directory
        base_sepsis_paths = [output_path / path_segment for path_segment in yaml_data["sepsis_summary"]]
        for base_sepsis_path in base_sepsis_paths:
            Path.mkdir(base_sepsis_path, exist_ok=True)
            for subdir in yaml_data["sepsis_summary_types"]:
                Path.mkdir(base_sepsis_path / subdir, exist_ok=True)
        # write general files
        # Save encounter summary
        UNIQUE_FILE_ID = f"{processor_assignment}_{year}"
        base_path = output_path / yaml_data["sepsis_summary"]
        pd.concat(appended_enc_summaries).to_csv(
            base_path / "encounter_summary" / f"encounters_summary_{UNIQUE_FILE_ID}.csv",
            index=True,
        )
        # Save error summary
        pd.DataFrame(error_list, columns=["csn", "error"]).to_csv(
            base_path / "error_summary" / f"error_list_{UNIQUE_FILE_ID}.csv",
            index=False,
        )
        # Save sepsis files
        pd.concat(appended_sofa_scores).to_csv(
            base_path / "sofa_summary" / f"sofa_summary_{UNIQUE_FILE_ID}.csv",
            index=False,
        )
        pd.concat(appended_sep3_time).to_csv(
            base_path / "sep3_summary" / f"sepsis3_summary_{UNIQUE_FILE_ID}.csv",
            index=False,
        )
        pd.concat(appended_sirs_scores).to_csv(
            base_path / "sirs_summary" / f"sirs_summary_{UNIQUE_FILE_ID}.csv",
            index=False,
        )
        pd.concat(appended_sep2_time).to_csv(
            base_path / "sep2_summary" / f"sepsis2_summary_{UNIQUE_FILE_ID}.csv",
            index=False,
        )
        # write comorbidity files
        # ICD10
        # pd.DataFrame.from_dict(ahrq_ICD10_dict).T.to_csv(output_path / 'em_sepsis_summary' / 'ahrq_ICD10_summary' / ('ahrq_ICD10_summary_'+ unique_file_id +'.csv'), index = True, index_label='csn')
        # pd.DataFrame.from_dict(elix_ICD10_dict).T.to_csv(output_path / 'em_sepsis_summary' / 'elix_ICD10_summary' / ('elix_ICD10_summary_'+ unique_file_id +'.csv'), index = True, index_label='csn')
        # pd.DataFrame.from_dict(quan_deyo_ICD10_dict).T.to_csv(base_path / "quan_deyo_ICD10_summary" / ("quan_deyo_ICD10_summary_" + UNIQUE_FILE_ID + ".csv"), index=True, index_label="csn",)
        # pd.DataFrame.from_dict(quan_elix_ICD10_dict).T.to_csv(base_path / "quan_elix_ICD10_summary" / ("quan_elix_ICD10_summary_" + UNIQUE_FILE_ID + ".csv"), index=True, index_label="csn",)
        # pd.DataFrame.from_dict(ccs_ICD10_dict).T.to_csv(output_path / 'em_sepsis_summary' / 'ccs_ICD10_summary' / ('ccs_ICD10_summary_'+ unique_file_id +'.csv'), index = True, index_label='csn')
        logging.info(
            f"MkDct- Time to create write encounter pickles for {year} was {time.perf_counter() - start_csn_creation} (s)"
        )
    except Exception as e:
        logging.error(f"MkDct- Could not find or open the pickle for year {year}: {e}")
