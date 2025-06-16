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

Changes by Mehak Arora:
     - Moved load_yaml function, sepsis3_summary, and sofa_summary functions to utils.py
     - Added error handling to the dictionary construction process.
     - Made the code modular for yearly pickle creation and supertable creation.
     - renamed variables to be more descriptive.
     - added comorbidity summary to the supertable.
     - documentation updates.
     - corrected the multiple task array implementation.

"""
import pickle
import time
import glob
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
logging.basicConfig(level=logging.INFO)

import sepyIMPORT as si
import sepyDICT as sd
import utils




###########################################################################
########## Import Data Frames and Create Yearly Pickle ####################
###########################################################################
def import_data_frames(yearly_instance, configs):
    """
    Imports data from a YAML structure and applies it to methods of a passed instance.
    Args:
        yearly_instance (sepyIMPORT): The instance whose methods will be called.
    """
    import_start_time = time.time()
    logging.info(
        "Sepy is currently reading flat files and importing them for analysis. Thank you for waiting."
    )
    for method_name, params in configs["yearly_instance"].items():
        data_type = method_name.split('_')[1]
        params['data_type'] = data_type
        yearly_instance.import_data(**params)
    
    logging.info(f"Sepy took {time.time() - import_start_time} (s) to create a yearly pickle.")

###########################################################################
############################# Make Supertables ###########################
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
        yearly_data_instance, encounter_csn, bed_unit_mapping, thresholds, dialysis_info, sepyDICTConfigs
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


def process_csn_with_summaries(
    csn, 
    count, 
    chunk_size, 
    year, 
    supertable_write_path, 
    bed_unit_mapping, 
    bounds, 
    dialysis_year, 
    import_instance
):
    """
    Process a single CSN with all its summaries
    
    Args:
        csn: The CSN to process
        count: The count of this CSN in the processing list
        chunk_size: Total number of CSNs in this chunk
        year: The year being processed
        supertable_write_path: Path to save the supertable
        bed_unit_mapping: Mapping of bed locations to units
        bounds: Threshold values for labs
        dialysis_year: Dialysis information for the year
        import_instance: The yearly data import instance
        
    Returns:
        tuple: A tuple containing all the summary dataframes or None values if errors occurred
    """
    result = {
        'sofa_summary': None,
        'sep3_summary': None,
        'sirs_summary': None, 
        'sep2_summary': None,
        'enc_summary': None,
        'comorbidity_summary': None,
        'error': None
    }
    
    try:
        logging.info(f"Sepy- Processing patient csn: {csn}, {count} of {chunk_size} for year {year}")
        instance = process_csn(csn, supertable_write_path, bed_unit_mapping, bounds, dialysis_year, import_instance)
        logging.info(f"Sepy- Instance created for csn: {csn}")
    except Exception as e:
        error_msg = str(e.args[0]) if e.args else str(e)
        logging.error(f"Sepy- Error in creating instance for csn {csn}: {error_msg}")
        result['error'] = [csn, error_msg]
        return result
    
    # Running summaries with error handling
    try:
        result['sofa_summary'] = utils.sofa_summary(csn, instance)
    except Exception as e:
        logging.error(f"Sepy- Error in Sofa Summary for csn {csn}: {e}")
    
    try:
        result['sep3_summary'] = utils.sepsis3_summary(csn, instance)
    except Exception as e:
        logging.error(f"Sepy- Error in Sepsis 3 Summary for csn {csn}: {e}")
    
    try:
        result['sirs_summary'] = utils.sirs_summary(csn, instance)
    except Exception as e:
        logging.error(f"Sepy- Error in SIRS Summary for csn {csn}: {e}")
    
    try:
        result['sep2_summary'] = utils.sepsis2_summary(csn, instance)
    except Exception as e:
        logging.error(f"Sepy- Error in Sepsis 2 Summary for csn {csn}: {e}")
    
    try:
        result['enc_summary'] = utils.enc_summary(instance)
    except Exception as e:
        logging.error(f"Sepy- Error in Encounter Summary for csn {csn}: {e}")
    
    try:
        result['comorbidity_summary'] = utils.comorbidity_summary(csn, instance)
    except Exception as e:
        logging.error(f"Sepy- Error in Comorbidity Summary for csn {csn}: {e}")
    
    logging.info(f"Sepy- Encounter {count} of {chunk_size} is complete!")
    return result


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process EMR data for a specific year')
    parser.add_argument('year', type=int, help='The year for which data is being processed')
    parser.add_argument('data_config', type=str, default='configurations/emory_config.yaml', help='Path to the data configuration file in YAML format')
    parser.add_argument('sepyIMPORT_config', type=str, default='configurations/import_config.yaml', help='Path to the sepyIMPORT configuration file in YAML format')
    parser.add_argument('sepyDICT_config', type=str, default='configurations/dict_config.yaml', help='Path to the sepyDICT configuration file in YAML format')
    parser.add_argument('num_processes', type=int, default=10, help='Number of processes to use')
    parser.add_argument('processor_assignment', type=int, help='Processor assignment')
    args = parser.parse_args()
    
    year = args.year
    dataConfig_path = args.data_config
    sepyIMPORTConfigs_path = args.sepyIMPORT_config
    sepyDICTConfigs_path = args.sepyDICT_config
    num_processes = args.num_processes
    processor_assignment = args.processor_assignment
    dataConfig = utils.load_yaml(dataConfig_path)
    sepyIMPORTConfigs = utils.load_yaml(sepyIMPORTConfigs_path)
    sepyDICTConfigs = utils.load_yaml(sepyDICTConfigs_path)
    logging.info(f"Sepy- The total number of processes: {num_processes}")
    logging.info(f"Sepy- The import year is: {year}")
    logging.info(f"Sepy- The processor assignment is: {processor_assignment}")
    
    #####################################################
    ######### Create Dictionary of File Paths ###########
    #####################################################

    #parent directory for all the flat files
    DATA_PATH = Path(dataConfig["data_path"])
    #path to the directory containing the grouping files, i.e., files that map component id to clinical features
    GROUPINGS_PATH = Path(dataConfig["groupings_path"])
    #path to the directory where the supertable pickles will be written
    SUPERTABLE_OUTPUT_PATH = Path(dataConfig["supertable_output_path"])
    #path to the directory where the yearly dictionaries will be written
    YEARLY_DICTIONARY_OUTPUT_PATH = Path(dataConfig["yearly_pickle_output_path"])
    YEARLY_DICTIONARY_FILE_NAME = os.path.join(YEARLY_DICTIONARY_OUTPUT_PATH, dataConfig["dataset_identifier"] + str(year) + ".pickle")

    paths = {}
    comorbidity_types = dataConfig["dictionary_paths"]["comorbidity_types"]
    grouping_types = dataConfig["dictionary_paths"]["grouping_types"]
    flatfile_types = dataConfig["dictionary_paths"]["flatfile_types"]
    combined_files = dataConfig["dictionary_paths"]["combined_files"]
    
    for comorbidity_type, comorbidity_file in comorbidity_types:
        try:
            paths[f"{comorbidity_type}"] = glob.glob(
                f"{GROUPINGS_PATH}/comorbidities/{comorbidity_file}"
            )[0]
        except IndexError:
            logging.error(f"Sepy- could not find comorbidity file for {comorbidity_type}")

    for type, grouping_path in grouping_types:
        try:
            paths[f"{type}"] = glob.glob(f"{GROUPINGS_PATH}/{grouping_path}")[0]
        except IndexError:
            logging.error(f"Sepy- could not find grouping file for {type}")

    for flatfile_type, flatfile_name in flatfile_types:
        try:
            paths[f"{flatfile_type}"] = glob.glob(
                f"{DATA_PATH}/{year}/{flatfile_name}"
            )[0]
        except IndexError:
            logging.error(f"Sepy- could not find flatfile type for {flatfile_name}")

    for combined_file in combined_files:
        try:
            paths[f"{combined_file[0]}"] = glob.glob(
                f"{DATA_PATH}/{combined_file[1]}"
            )[0]
        except IndexError:
            logging.error(f"Sepy- could not find combined file for {combined_file[1]}")

    #####################################################
    ############ Create Pickle of Yearly Data ###########
    #####################################################
    if dataConfig["make_yearly_pickle"] == "yes":
        try:
            start = time.perf_counter()
            logging.info(f"Creating yearly pickle for {year}")
            logging.info(f"Yearly pickle will be saved to {YEARLY_DICTIONARY_FILE_NAME}")

            import_instance = si.sepyIMPORT(paths, sepyIMPORTConfigs)
            logging.info(f"An instance of the sepyIMPORT class was created for {year}")
            
            logging.info(f"Importing data frames for {year}")
            logging.info(f"This may take a while...")
            import_data_frames(import_instance, dataConfig)
            logging.info(f"Data frames imported for {year}")
            logging.info(f"Dumping import instance to pickle for {year}")

            with open(YEARLY_DICTIONARY_FILE_NAME, "wb") as handle:
                pickle.dump(import_instance, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f"Yearly pickle for {year} created and saved")
            logging.info(
                f"Time to create {year}s data and write to pickles was {time.perf_counter()-start} (s)"
            )
        except (FileNotFoundError, ValueError, KeyError) as e:
            logging.error(e)
    else: #If the yearly pickle already exists, read it in
        logging.info(f"Skipping creation of yearly pickle for {year} -  it should already exist")
        if os.path.exists(YEARLY_DICTIONARY_FILE_NAME):
            logging.info(f"Yearly pickle for {year} exists")
        else:
            logging.error(f"Yearly pickle for {year} does not exist. Please check the config file.")
            exit()

        #Read the yearly pickle
        try:
            with open(YEARLY_DICTIONARY_FILE_NAME, "rb") as handle:
                import_instance = pickle.load(handle)
        except Exception as e:
            logging.error(f"Error loading yearly pickle for {year}: {e}")
            exit()
        
    ###########################################################################
    ###################### Begin Dictionary Construction ######################
    ###########################################################################
    if dataConfig["make_supertables"] == "yes":
        try:
            # set file path for unique encounters to create supertables for 
            encounters_path = dataConfig["encounters_path"]
            if encounters_path == "ENCOUNTER":
                encounters_path = paths["ENCOUNTER"]
            logging.info(f"Sepy- The encounters path: {encounters_path}")
            
            # set filters for encounters
            encounter_type = dataConfig["encounter_type"]
            age = dataConfig["age"]
            specific_enc_filter = dataConfig["specific_enc_filter"]
            specific_enc = dataConfig["specific_enc"]

            # reads the list of csns
            csn_df = pd.read_csv(encounters_path, sep = "|", header = 0)
            num_encounters = len(csn_df)
            logging.info(f"Sepy- The year {year} has {num_encounters} encounters before filtering.")
        except (IndexError, ValueError, TypeError) as e:
            logging.error(
                f"Sepy- There was an error importing one of the arguments: {type(e).__name__}."
            )
            logging.info(f"Sepy- You are trying to load the following CSN list {year}")

        ###########################################################################
        ############ Filter the encounters based on configs #######################
        ###########################################################################
        
        # If specific encounter filter is applied, filter the encounters based on the list of specific encounters in the config file
        if specific_enc_filter == "yes":
            if "specific_enc_filter_list" in dataConfig and dataConfig["specific_enc_filter_list"] and os.path.exists(dataConfig["specific_enc_filter_list"]) and dataConfig["specific_enc_filter_list"].endswith('.csv'):
                try:
                    specific_enc_filter_list = pd.read_csv(dataConfig["specific_enc_filter_list"])
                except Exception as e:
                    logging.error(f"Sepy- Error in the specified encounter filter list. Please check the config file. {e}")
                try:
                    csn_df = csn_df[csn_df.csn.isin(specific_enc_filter_list["csn"])]
                except Exception as e:
                    logging.error(f"Sepy- Error in filtering encounters. Please check the config file. {e}")
                
                num_encounters = len(csn_df)
                logging.info(f"Sepy- The year {year} has {num_encounters} encounters after filtering.")
            else:
                logging.info(f"Sepy- Error in the specified encounter filter list. Please check the config file.")
        else:
            logging.info(f"Sepy- No specific encounter filter was applied")
            
        # If encounter type filter is applied, filter the encounters based on the encounter type in the config file (EM, IN, all)
        if encounter_type != "all":
            csn_df = csn_df[csn_df.encounter_type == encounter_type]
            num_encounters = len(csn_df)
            logging.info(f"Sepy- The year {year} has {num_encounters} encounters after filtering.")
        else:
            logging.info(f"Sepy- No specific encounter type filter was applied")
        
        # If age filter is applied, filter the encounters based on the age in the config file (adult, pediatric, all)
        if age == "adult":
            csn_df = csn_df[csn_df.age >= 18]
            num_encounters = len(csn_df)
            logging.info(f"Sepy- The year {year} has {num_encounters} encounters after filtering.")
        elif age == "pediatric":
            csn_df = csn_df[csn_df.age < 18]
            num_encounters = len(csn_df)
            logging.info(f"Sepy- The year {year} has {num_encounters} encounters after filtering.")
        else:
            logging.info(f"Sepy- No specific age filter was applied")
        
        # drop duplicates
        csn_df = csn_df.drop_duplicates()
        total_num_enc = len(csn_df)
        logging.info(f"Sepy- The year {year} has {total_num_enc} encounters after filtering and dropping duplicates.")

        
        ################################################
        ############ Create Chunks of Encounters #######
        ################################################
        chunk_size = int(total_num_enc / num_processes)
        logging.info(f"Sepy- The ~chunk size is {chunk_size}")
        
        list_of_chunks = np.array_split(csn_df, num_processes)
        logging.info(f"Sepy- The list of chunks has {len(list_of_chunks)} unique dataframes.")
        
        # uses processor assignment to select correct chunk
        process_list = list_of_chunks[processor_assignment]["csn"]
        logging.info(f"Sepy- The process_list head:\n {process_list.head()}")
        
        # select correct pickle by year
        pickle_name = YEARLY_DICTIONARY_OUTPUT_PATH + (dataConfig["dataset_identifier"] + str(year) + ".pickle")
        logging.info(f"Sepy- The following pickle is being read: {pickle_name}")
        
        supertable_write_path = SUPERTABLE_OUTPUT_PATH / str(year)
        supertable_write_path.mkdir(exist_ok = True)
        logging.info(f"Sepy-Directory for year {year} was set to {supertable_write_path}")
        
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
            logging.error("Sepy- The bed to unit mapping file is not formatted correctly.")
        
        dialysis = pd.read_csv(paths["DIALYSIS"])   
        dialysis_year = dialysis.loc[
            dialysis["Encounter Number"].isin(csn_df["csn"].values)
        ]

        bounds = pd.read_csv(VARIABLE_BOUNDS_CSV_FNAME)

        ###########################################################################
        ######################### Make Dicts by CSN ###############################
        ###########################################################################
        logging.info("making dicts")
        
        # Determine the number of workers for local multiprocessing
        # Use a smaller number of local processes to avoid overloading the system
        # since we're already running multiple SLURM jobs
        num_local_workers = min(os.cpu_count(), 4)  # Use at most 4 cores per SLURM job
        logging.info(f"Sepy- Using {num_local_workers} local worker processes")
        
        # Create partial function with fixed arguments
        process_func = partial(
            process_csn_with_summaries,
            chunk_size=len(process_list),
            year=year,
            supertable_write_path=supertable_write_path,
            bed_unit_mapping=bed_to_unit_mapping,
            bounds=bounds,
            dialysis_year=dialysis_year,
            import_instance=import_instance
        )
        
        # Initialize empty lists for results
        appended_sofa_scores = []
        appended_sep3_time = []
        appended_sirs_scores = []
        appended_sep2_time = []
        appended_enc_summaries = []
        appended_comorbidity_summaries = []
        
        # Process CSNs in parallel
        with ProcessPoolExecutor(max_workers=num_local_workers) as executor:
            # Submit all tasks
            future_to_csn = {
                executor.submit(process_func, csn, i+1): (csn, i+1) 
                for i, csn in enumerate(process_list)
            }
            
            # Process results as they complete
            for future in as_completed(future_to_csn):
                csn, count = future_to_csn[future]
                try:
                    result = future.result()
                    if result['error']:
                        error_list.append(result['error'])
                        continue
                        
                    # Append valid results to their respective lists
                    if result['sofa_summary'] is not None:
                        appended_sofa_scores.append(result['sofa_summary'])
                    if result['sep3_summary'] is not None:
                        appended_sep3_time.append(result['sep3_summary'])
                    if result['sirs_summary'] is not None:
                        appended_sirs_scores.append(result['sirs_summary'])
                    if result['sep2_summary'] is not None:
                        appended_sep2_time.append(result['sep2_summary'])
                    if result['enc_summary'] is not None:
                        appended_enc_summaries.append(result['enc_summary'])
                    if result['comorbidity_summary'] is not None:
                        appended_comorbidity_summaries.append(result['comorbidity_summary'])
                        
                except Exception as e:
                    logging.error(f"Sepy- Error processing result for csn {csn}: {e}")
                    error_list.append([csn, str(e)])

        ###########################################################################
        ########################## Export Sepsis Summary ##########################
        ###########################################################################
        # create sepsis_summary directory
        base_sepsis_path = SUPERTABLE_OUTPUT_PATH / dataConfig["sepsis_summary"] / str(year)
        Path.mkdir(base_sepsis_path, exist_ok=True)
        for subdir in dataConfig["sepsis_summary_types"]:
            Path.mkdir(base_sepsis_path / subdir, exist_ok=True)

        # Save encounter summary
        UNIQUE_FILE_ID = f"{processor_assignment}_{year}"
        base_path = SUPERTABLE_OUTPUT_PATH / dataConfig["sepsis_summary"] / str(year)
        
        # Check if any results were collected before trying to concatenate
        if appended_enc_summaries:
            pd.concat(appended_enc_summaries).to_csv(
                base_path / "encounter_summary" / f"encounters_summary_{UNIQUE_FILE_ID}.csv",
                index=True,
            )
        else:
            logging.warning("Sepy- No encounter summaries were collected")
            
        # Save comorbidity summary
        if appended_comorbidity_summaries:
            pd.concat(appended_comorbidity_summaries).to_csv(
                base_path / "comorbidity_summary" / f"comorbidity_summary_{UNIQUE_FILE_ID}.csv",
                index=False,
            )
        else:
            logging.warning("Sepy- No comorbidity summaries were collected")

        # Save error summary
        pd.DataFrame(error_list, columns=["csn", "error"]).to_csv(
            base_path / "error_summary" / f"error_list_{UNIQUE_FILE_ID}.csv",
            index=False,
        )
        
        # Save sepsis files
        if appended_sofa_scores:
            pd.concat(appended_sofa_scores).to_csv(
                base_path / "sofa_summary" / f"sofa_summary_{UNIQUE_FILE_ID}.csv",
                index=False,
            )
        else:
            logging.warning("Sepy- No SOFA scores were collected")
            
        if appended_sep3_time:
            pd.concat(appended_sep3_time).to_csv(
                base_path / "sep3_summary" / f"sepsis3_summary_{UNIQUE_FILE_ID}.csv",
                index=False,
            )
        else:
            logging.warning("Sepy- No Sepsis-3 summaries were collected")
            
        if appended_sirs_scores:
            pd.concat(appended_sirs_scores).to_csv(
                base_path / "sirs_summary" / f"sirs_summary_{UNIQUE_FILE_ID}.csv",
                index=False,
            )
        else:
            logging.warning("Sepy- No SIRS scores were collected")
            
        if appended_sep2_time:
            pd.concat(appended_sep2_time).to_csv(
                base_path / "sep2_summary" / f"sepsis2_summary_{UNIQUE_FILE_ID}.csv",
                index=False,
            )
        else:
            logging.warning("Sepy- No Sepsis-2 summaries were collected")
        
        logging.info(
            f"Sepy- Time to create write encounter pickles for {year} was {time.perf_counter()-start_csn_creation}s"
        )