# -*- coding: utf-8 -*-
"""
This pipeline provides functions to import data from flat files into pandas dataframes.
The dataframes are then pickled for use in super table construction.

Elite Data Hacks
Author: Christopher S. Josef, MD
Email: csjosef@krvmail.com
Version: 0.1

Kameleswaran Labs
Author: Jack F. Regan
Edited: 025-03-01
Vtrsion: 0.2

Changes:
     - combined import and dictionary construction into a single script.
     - update documentation.
     - added configuration file through yaml for extensibility.
"""
import pickle
import re
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

logging.basicConfig(level=logging.INFO)
###########################################################################
############################ File Dictionaries ############################
###########################################################################
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
        paths[f"{comorbidity}"] = glob.glob(
            f"{GROUPINGS_PATH}/comorbidities/{comorbidity}.csv"
        )[0]
    for type in grouping_types:
        paths[f"{type}"] = glob.glob(f"{GROUPINGS_PATH}/{type}*.csv")[0]
    for year_type in year_types:
        paths[f"{year_type[0]}"] = glob.glob(
            f"{DATA_PATH}/{data_year}/*{year_type[1]}*.dsv"
        )[0]
    return paths
###########################################################################
########################### Import Data Frames ############################
###########################################################################
def import_data_frames(yearly_instance):
    """
    Imports data from a YAML structure and applies it to methods of a passed instance.
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
            # check if method requires numeric_cols parameter and access list in sepyIMPORT instnace
            if "numeric_cols" in params and isinstance(params["numeric_cols"], str):
                params["numeric_cols"] = getattr(yearly_instance, params["numeric_cols"], None)
        method(**params)
    logging.info(f"Sepy took {time.time() - import_start_time} (s) to create a yearly pickle.")
###########################################################################
############################ Make Dictionaries ############################
###########################################################################
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
        encounter_csn (str): The unique encounter ID (CSN) to process.
        pickle_save_path (Path): The directory path where the pickle file will be saved.
        bed_unit_mapping (dict): A mapping of bed locations to ICU units.
        thresholds (dict): A dictionary containing threshold values or limits used in processing.
        dialysis_info (dict): Information related to dialysis treatment for the patient.
        yearly_data_instance (object): An instance of the `sepyIMPORT` class containing the yearly data.
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
###########################################################################
############################# Summary Functions ###########################
###########################################################################
def sofa_summary(encounter_csn, encounter_instance):
    """
    Summarizes the SOFA scores for a single patient encounter and appends the data to the global list.

    Args:
        encounter_csn (str): The unique encounter ID (CSN) for the patient encounter.
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
        encounter_csn (str): The unique encounter ID (CSN) for the patient encounter.
        encounter_instance (sepyDICT): An instance of the sepyDICT class, containing the encounter data.
    """
    sep3_time = encounter_instance.encounter_dict["sep3_time"]
    sep3_time["csn"] = encounter_csn  # add csn to sep3 time
    appended_sep3_time.append(sep3_time)
def sirs_summary(encounter_csn, encounter_instance):
    """
    Summarizes the SIRS scores for a single patient encounter and appends the data to the global list.

    Args:
        encounter_csn (str): The unique encounter ID (CSN) for the patient encounter.
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
        encounter_csn (str): The unique encounter ID (CSN) for the patient encounter.
        encounter_instance (sepyDICT): An instance of the sepyDICT class, containing the encounter data.
    """
    sep2_time = encounter_instance.encounter_dict["sep2_time"]
    sep2_time["csn"] = encounter_csn  # add csn to sep3 time
    appended_sep2_time.append(sep2_time)
def enc_summary(encounter_instance):
    """
    Summarizes encounter-level data by combining flags, static features, and event times, then appends it to the global list.

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
    appended_enc_summaries.append(enc_summary_df)
def comorbidity_summary(encounter_csn, encounter_instance):
    """
    Summarizes the comorbidity data for a single patient encounter based on a configuration file.

    Args:
        encounter_csn (str): The unique encounter ID (CSN) for the patient encounter.
        encounter_instance (sepyDICT): An instance of the sepyDICT class, containing comorbidity-related data.
    """
    for summary_name in yaml_data['comorbidity_summary']:
        try:
            comorbidity_summary_dicts[summary_name + '_dict'][encounter_csn] = getattr(encounter_instance, f"{summary_name}_PerCSN").icd_count
        except AttributeError:
            logging.warning(f"Attribute {summary_name}_PerCSN not found for csn {encounter_csn}")
        except KeyError as e:
            logging.error(f"Key error for {summary_name}_dict: {e}")
        except Exception as e:
            logging.error(f"Error processing comorbidity {summary_name} for csn {encounter_csn}: {e}")
###########################################################################
################################ Load YAML ################################
###########################################################################
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
###########################################################################
################################ File Paths ###############################
###########################################################################
# Generate file paths
### data path is the parent directory for all the flat files; you'll specify each file location below
DATA_PATH = yaml_data["data_path"]
### grouping path is where the lists of meds, labs, & comorbs will be located
GROUPINGS_PATH = yaml_data["groupings_path"]
### output paths is where the pickles will be written
PICKLE_OUTPUT_PATH = yaml_data["pickle_output_path"]
### bed unit csv is a mapping of bed units to icu type [ed, ward, icu]
BED_UNIT_CSV_FNAME = yaml_data["bed_unit_csv_fname"]
### TODO: add variable bounds csv description
VARIABLE_BOUNDS_CSV_FNAME = Path(yaml_data["variable_bounds_csv_fname"])
### dialysis info csn is a mapping of dialysis info to csn
DIALYSIS_INFO_CSN_FNAME = Path(yaml_data["dialysis_info_csn_fname"])
### dictionary output path is where the dictionaries will be written
DICTIONARY_OUTPUT_PATH = Path(yaml_data["dictionary_output_path"])
###########################################################################
##################### Initialize Empty Summaries ##########################
###########################################################################
comorbidity_summary_dicts = {}

for summary_name in yaml_data['comorbidity_summary']:
    comorbidity_summary_dicts[summary_name + '_dict'] = {}
# other summaries
appended_sofa_scores = []
appended_sep3_time = []
appended_sirs_scores = []
appended_sep2_time = []
appended_enc_summaries = []

start = time.perf_counter()
###########################################################################
############################## Main Function ##############################
###########################################################################
if __name__ == "__main__":
    # Usage:
    #   python make_dicts.py <year> <CONFIGURATION_PATH>
    # Parameters:
    #   <year> (int): The year for which data is being processed.
    #   <CONFIGURATION_PATH> (str): Path to the configuration file in YAML format.
    year = int(sys.argv[1])
    if yaml_data["make_pickle"] == "yes":
        try:
            # Generate paths for each year dynamically and store them in a dictionary
            path_dictionary = {}
            for year_to_load in yaml_data["years_to_scrape_data"]:
                path_dictionary[int(year_to_load)] = generate_paths(year_to_load)
            start = time.perf_counter()
            # creates pickle file name
            PICKLE_FILE_NAME = PICKLE_OUTPUT_PATH + yaml_data["dataset_identifier"] + str(year) + ".pickle"
            logging.info(PICKLE_FILE_NAME)
            # creates path dictionary name
            PATH_DICTIONARY = "path_dictionary" + str(year)
            logging.info(f"File locations were taken from the path dictionary: {PATH_DICTIONARY}")
            import_instance = si.sepyIMPORT(path_dictionary[year], "|", yaml_data)
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
###########################################################################
###################### Begin Dictionary Construction ######################
###########################################################################
    if yaml_data["make_dictionary"] == "yes":
        try:
            logging.info(sys.version)
            # set file path for unique encounters
            unique_encounters_path = yaml_data["unique_encounters_path"]
            logging.info(f"MkDct- The unique encounters path: {unique_encounters_path}")
            # set filter for unique encounters
            encounter_type = yaml_data["encounter_type"]
            age = yaml_data["age"]
            unique_enc_filter = yaml_data["unique_enc_filter"]
            unique_enc = yaml_data["unique_enc"]
            # sets directoy for yearly pickles
            pickle_path = PICKLE_OUTPUT_PATH
            logging.info(f"MkDct- The pickle directory: {PICKLE_OUTPUT_PATH}")
            # sets directory for output
            output_path = DICTIONARY_OUTPUT_PATH
            logging.info(f"MkDct- The output directory: {DICTIONARY_OUTPUT_PATH}")
            # total number of tasks/processes
            num_processes = int(yaml_data["num_processes"])
            logging.info(f"MkDct- The tot num of processes: {num_processes}")
            # task number
            processor_assignment = int(yaml_data["processor_assignment"])
            logging.info(f"MkDct- This task is: {processor_assignment}")
            # year num provided in bash
            logging.info(f"MkDct- The import year is: {year}")
        except (IndexError, ValueError, TypeError) as e:
            logging.error(
                f"MkDct- There was an error importing one of the arguments: {type(e).__name__}."
            )
            logging.info(f"MkDct- You are trying to load the following CSN list {year}")
            logging.info(f"MkDct- You are trying to use this many processors {num_processes}")
###########################################################################
########### Create Encounter List Based on Processor Assignment ###########
###########################################################################
        # Cohort selector
        ed = 0
        in_pt = 1
        icu = 0
        adult = 1
        vent_row = 0
        vent_start = 0
        # reads the list of csns
        csn_df = pd.read_csv(unique_encounters_path, sep = "|", header = 0)
        # checks if the specific encounter type filter is applied
        if unique_enc_filter == "yes":
            csn_df = csn_df[csn_df.csn.isin(unique_enc_filter)]
        else:
            logging.info(f"MkDct- No specific encounter type filter was applied")
            if encounter_type != "all":
                csn_df = csn_df[csn_df.encounter_type == encounter_type]
            logging.info(f"MkDct- The encounter type filter was applied: {encounter_type}")
            if age == "adult":
                csn_df = csn_df[csn_df.age >= 18]
            elif age == "pediatric":
                csn_df = csn_df[csn_df.age < 18]
            logging.info(f"MkDct- The age filter was applied: {age}")
        # drop duplicates
        csn_df = csn_df.drop_duplicates()
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
        pickle_name = pickle_path + (yaml_data["dataset_identifier"] + str(year) + ".pickle")
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
            dict_write_path = output_path / str(year)
            dict_write_path.mkdir(exist_ok = True)
            logging.info(f"MkDct-Directory for year {year} was set to {dict_write_path}")
            # make empty list to handle csn's with errors
            error_list = []
###########################################################################
#################### Load Files for Extra Processing ######################
###########################################################################
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
            dialysis = pd.read_csv(DIALYSIS_INFO_CSN_FNAME)
            bounds = pd.read_csv(VARIABLE_BOUNDS_CSV_FNAME)
            logging.info(dialysis.head())
            dialysis_year = dialysis.loc[
                dialysis["Encounter Number"].isin(csn_df["csn"].values)
            ]
###########################################################################
######################### Make Dicts by CSN ###############################
###########################################################################
            logging.info("making dicts")
            list=[1,2,1,2,2,2,2,2]
            for count, csn in enumerate(process_list, start=1):
                try:
                    logging.info(f"MkDct- Processing patient csn: {csn}, {count} of {chunk_size} for year {year}")
                    instance = process_csn(csn, dict_write_path, bed_to_unit_mapping, bounds, dialysis_year, yearly_instance)
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
###########################################################################
########################## Export Sepsis Summary ##########################
###########################################################################
            # create sepsis_summary directory
            base_sepsis_path = output_path / yaml_data["sepsis_summary"] / str(year)
            Path.mkdir(base_sepsis_path, exist_ok=True)
            for subdir in yaml_data["sepsis_summary_types"]:
                Path.mkdir(base_sepsis_path / subdir, exist_ok=True)
            # write general files
            # Save encounter summary
            UNIQUE_FILE_ID = f"{processor_assignment}_{year}"
            base_path = output_path + yaml_data["sepsis_summary"] / str(year)
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
            # pd.DataFrame.from_dict(ICD10_ahrq_dict).T.to_csv(base_path / 'ICD10_ahrq_summary' / ('ICD10_ahrq_summary_'+ unique_file_id +'.csv'), index = True, index_label='csn')
            # pd.DataFrame.from_dict(ICD10_elix_dict).T.to_csv(base_path / 'ICD10_elix_summary' / ('ICD10_elix_summary_'+ unique_file_id +'.csv'), index = True, index_label='csn')
            # pd.DataFrame.from_dict(ICD10_quan_deyo_dict).T.to_csv(base_path / "ICD10_quan_deyo_summary" / ("ICD10_quan_deyo_summary_" + UNIQUE_FILE_ID + ".csv"), index=True, index_label="csn",)
            # pd.DataFrame.from_dict(ICD10_quan_elix_dict).T.to_csv(base_path / "ICD10_quan_elix_summary" / ("ICD10_quan_elix_summary_" + UNIQUE_FILE_ID + ".csv"), index=True, index_label="csn",)
            # pd.DataFrame.from_dict(ICD10_single_ccs_dict).T.to_csv(base_path / 'ICD10_single_ccs_summary' / ('ICD10_single_ccs_summary_'+ unique_file_id +'.csv'), index = True, index_label='csn')
            logging.info(
                f"MkDct- Time to create write encounter pickles for {year} was {time.perf_counter()-start_csn_creation}s"
            )
        except Exception as e:
            logging.error(f"MkDct- Could not find or open the pickle for year {year}: {e}")