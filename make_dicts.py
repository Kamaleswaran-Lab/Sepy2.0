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
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
logging.basicConfig(level=logging.INFO)

import sys
sys.path.append("/hpc/home/ma618/Sepy/")

import sepyIMPORT as si
import sepyDICT as sd
import utils



###########################################################################
############################# Make Supertables ###########################
###########################################################################
def make_sepyMaster(yearly_data_instance, sepyConfigs, bounds, save_dir):
    """
    Creates a sepyMaster instance for a given year.
    
    Args:
        yearly_data_instance (object): An instance of the `sepyIMPORT` class containing the yearly data.
        sepyConfigs (dict): A dictionary containing the configuration settings for the sepyDICT class.  
        bounds (dict): A dictionary containing the threshold values for the supertable.
        save_dir (str): The directory where the supertable will be saved.
    Returns:
        sepyMaster: An instance of the `sepyMaster` class containing the processed encounter data.
    """
    sepyMaster_instance = sd.sepyMaster(yearly_data_instance, sepyConfigs, bounds, save_dir)
    return sepyMaster_instance


def process_csn_instance(
    csn_instance,
    count,
    chunk_size,
    year
):
    """
    Process a pre-created sepyCSN instance.
    
    Args:
        csn_instance: Pre-created sepyCSN instance
        count: Current count in processing
        chunk_size: Total number of CSNs to process
        year: Year being processed
    Returns:
        dict: Dictionary containing all summary dataframes or None values if errors occurred
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
        logging.info(f"Sepy- Processing patient csn: {csn_instance.clinical_data.csn}, {count} of {chunk_size} for year {year}")
        csn_instance.process()
        logging.info(f"Sepy- Instance processed for csn: {csn_instance.clinical_data.csn}")
    except Exception as e:
        error_msg = str(e.args[0]) if e.args else str(e)
        logging.error(f"Sepy- Error in processing instance for csn {csn_instance.clinical_data.csn}: {error_msg}")
        result['error'] = [csn_instance.clinical_data.csn, error_msg]
        return result
    
    # Running summaries with error handling
    try:
        result['sofa_summary'] = utils.sofa_summary(csn_instance)
    except Exception as e:
        logging.error(f"Sepy- Error in Sofa Summary for csn {csn_instance.clinical_data.csn}: {e}")
    
    try:
        result['sep3_summary'] = utils.sepsis3_summary(csn_instance)
    except Exception as e:
        logging.error(f"Sepy- Error in Sepsis 3 Summary for csn {csn_instance.clinical_data.csn}: {e}")
    
    try:
        result['sirs_summary'] = utils.sirs_summary(csn_instance)
    except Exception as e:
        logging.error(f"Sepy- Error in SIRS Summary for csn {csn_instance.clinical_data.csn}: {e}")
    
    
    try:
        result['enc_summary'] = utils.enc_summary(csn_instance)
    except Exception as e:
        logging.error(f"Sepy- Error in Encounter Summary for csn {csn_instance.clinical_data.csn}: {e}")
    
    try:
        series_dict = {k.replace('_dict', ''): pd.Series(v) for k, v in utils.comorbidity_summary(csn_instance, dataConfig).items()}
        df = pd.DataFrame(series_dict).reset_index()
        df.rename(columns={'index': 'csn'}, inplace=True)
        result['comorbidity_summary'] = df
    except Exception as e:
        logging.error(f"Sepy- Error in Comorbidity Summary for csn {csn_instance.clinical_data.csn}: {e}")
    
    logging.info(f"Sepy- Encounter {count} of {chunk_size} is complete!")
    return result

def process_batch_of_csns(process_list, sepyMaster_instance, year, start_count):
    """
    Process a batch of CSNs and return their results
    
    Args:
        process_list: List of CSNs to process
        sepyMaster_instance: Instance of sepyMaster
        year: Year being processed
        start_count: Starting count for this batch
    Returns:
        list: List of results for each CSN in the batch
    """
    results = []
    num_local_workers = min(os.cpu_count() - 1, 4)
    
    # Create CSN instances for just this batch
    csn_instances = []
    for i, csn in enumerate(process_list):
        csn_instance = sepyMaster_instance.create_csn_instance(csn)
        csn_instances.append(csn_instance)

    # Process the batch in parallel
    with ProcessPoolExecutor(max_workers=num_local_workers) as executor:
        future_to_csn = {
            executor.submit(
                process_csn_instance,
                csn_instance=csn_instance,
                count=start_count+i,
                chunk_size=len(process_list),
                year=year
            ): (csn_instance.clinical_data.csn, start_count+i) 
            for i, csn_instance in enumerate(csn_instances)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_csn):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                csn, _ = future_to_csn[future]
                logging.error(f"Sepy- Error processing result for csn {csn}: {e}")
                results.append({
                    'error': [csn, str(e)],
                    'sofa_summary': None,
                    'sep3_summary': None,
                    'sirs_summary': None,
                    'sep2_summary': None,
                    'enc_summary': None,
                    'comorbidity_summary': None
                })
    
    # Clear the batch from memory
    del csn_instances
    gc.collect()
    
    return results


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process EMR data for a specific year')
    parser.add_argument('--year', type=int, help='The year for which data is being processed')
    parser.add_argument('--data_config', type=str, default='configurations/emory_config.yaml', help='Path to the data configuration file in YAML format')
    parser.add_argument('--sepy_config', type=str, default='configurations/dict_config.yaml', help='Path to the sepyIMPORT configuration file in YAML format')
    parser.add_argument('--num_processes', type=int, default=10, help='Number of processes to use')
    parser.add_argument('--processor_assignment', default=0, type=int, help='Processor assignment')
    args = parser.parse_args()
    
    year = args.year
    dataConfig_path = args.data_config
    sepyConfigs_path = args.sepy_config
    num_processes = args.num_processes
    processor_assignment = args.processor_assignment
    dataConfig = utils.load_yaml(dataConfig_path)
    sepyConfigs = utils.load_yaml(sepyConfigs_path)
    logging.info(f"Sepy- The total number of processes: {num_processes}")
    logging.info(f"Sepy- The import year is: {year}")
    logging.info(f"Sepy- The processor assignment is: {processor_assignment}")
    
    #####################################################
    ######### Create Dictionary of File Paths ###########
    #####################################################
    

    DATA_PATH = Path(os.path.expandvars(dataConfig["data_path"]))
    #path to the directory containing the grouping files, i.e., files that map component id to clinical features
    GROUPINGS_PATH = Path(os.path.expandvars(dataConfig["groupings_path"]))
    #path to the directory where the supertable pickles will be written
    SUPERTABLE_OUTPUT_PATH = Path(os.path.expandvars(dataConfig["supertable_output_path"]))
    #path to the directory where the yearly dictionaries will be written
    YEARLY_DICTIONARY_OUTPUT_PATH = Path(os.path.expandvars(dataConfig["yearly_pickle_output_path"]))
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
                raise Exception(f"Sepy- could not find comorbidity file for {comorbidity_type}")

    for type, grouping_path in grouping_types:
        try:
            paths[f"{type}"] = glob.glob(f"{GROUPINGS_PATH}/{grouping_path}")[0]
        except IndexError:
            logging.error(f"Sepy- could not find grouping file for {type}")
            raise Exception(f"Sepy- could not find grouping file for {type}")

    for flatfile_type, flatfile_name in flatfile_types:
        try:
            paths[f"{flatfile_type}"] = glob.glob(
                f"{DATA_PATH}/{year}/{flatfile_name}"
            )[0]
        except IndexError:
            logging.error(f"Sepy- could not find flatfile type for {flatfile_name}")
            raise Exception(f"Sepy- could not find flatfile type for {flatfile_name}")

    for combined_file in combined_files:
        try:
            paths[f"{combined_file[0]}"] = glob.glob(
                f"{DATA_PATH}/*{combined_file[1]}*"
            )[0]
            
        except IndexError:
            logging.error(f"Sepy- could not find combined file for {combined_file[1]}")
            raise Exception(f"Sepy- could not find combined file for {combined_file[1]}")

    file_dictionary = paths

    #####################################################
    ############ Create Pickle of Yearly Data ###########
    #####################################################
    if dataConfig["make_yearly_pickle"] == "yes":
        try:
            start = time.perf_counter()
            logging.info(f"Creating yearly pickle for {year}")
            logging.info(f"Yearly pickle will be saved to {YEARLY_DICTIONARY_FILE_NAME}")

            import_instance = si.sepyIMPORT(paths, sepyConfigs, dataConfig["yearly_instance"])
            logging.info(f"An instance of the sepyIMPORT class was created for {year}, data was automatically imported")
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
            # set file path for the file containing the list of unique encounters to create supertables for 
            # ENCOUNTER sets it to the path of the encounter flatfile 
            encounters_path = dataConfig["encounters_path"]
            if encounters_path == "ENCOUNTER_FILE":
                encounters_path = paths["ENCOUNTER"]
            logging.info(f"Sepy- The encounters path: {encounters_path}")

            # set filters for encounters 
            encounter_type = dataConfig["encounter_type"] #IN, EM, all
            age = dataConfig["age"] #adult, pediatric, all

            # set filters for specific encounter numbers (csns)
            specific_enc_filter = dataConfig["specific_enc_filter"] #yes, no

            # reads the list of csns from the encounters path
            # csn_df = pd.read_csv(encounters_path, sep = "|")
            csn_df = import_instance.df_encounters
            num_encounters = len(csn_df)
            logging.info(f"Sepy- The year {year} has {num_encounters} encounters before filtering.")
        except (IndexError, ValueError, TypeError) as e:
            logging.error(
                f"Sepy- There was an error importing one of the arguments: {type(e).__name__}."
            )
            

        ###########################################################################
        ############ Filter the encounters based on configs #######################
        ###########################################################################
        
        # If specific encounter filter is applied, filter the encounters based on the list of specific encounters in the config file
        if specific_enc_filter == "yes":
            if "specific_enc_filter_list" in dataConfig and dataConfig["specific_enc_filter_list"]:
                
                try:
                    if os.path.exists(dataConfig["specific_enc_filter_list"]) and dataConfig["specific_enc_filter_list"].endswith('.csv'):
                        specific_enc_filter_list = pd.read_csv(dataConfig["specific_enc_filter_list"])
                        try:
                            csn_df = csn_df[csn_df.index.isin(specific_enc_filter_list["csn"])]
                        except Exception as e:
                            logging.error(f"Sepy- Error in filtering encounters. Please check the config file. {e}")
                except Exception as e:
                    logging.error(f"Sepy- Error in the specified encounter filter list. List is not a csv file. {e}")
                # if the list is a list in the config file, convert it to a dataframe
                    if isinstance(dataConfig["specific_enc_filter_list"], list):
                        specific_enc_filter_list = pd.DataFrame(dataConfig["specific_enc_filter_list"], columns=["csn"])
                        try:
                            csn_df = csn_df[csn_df.index.isin(specific_enc_filter_list["csn"])]
                        except Exception as e:
                            logging.error(f"Sepy- Error in filtering encounters. Please check the config file. {e}")
            else:
                logging.info(f"Sepy- Error in the specified encounter filter list. Please check the config file.")
        else:
            logging.info(f"Sepy- No specific encounter filter was applied")

        num_encounters = len(csn_df)
        logging.info(f"Sepy- The year {year} has {num_encounters} encounters after filtering.")
            
        # If encounter type filter is applied, filter the encounters based on the encounter type in the config file (EM, IN, all)
        if encounter_type != "all":
            csn_df = csn_df[csn_df["encounter_type"] == encounter_type]
            num_encounters = len(csn_df)
            logging.info(f"Sepy- The year {year} has {num_encounters} encounters after filtering.")
        else:
            logging.info(f"Sepy- No specific encounter type filter was applied")
        
        # If age filter is applied, filter the encounters based on the age in the config file (adult, pediatric, all)
        if age == "adult":
            csn_df = csn_df[csn_df.age >= 18]
            num_encounters = len(csn_df)
            logging.info(f"Sepy- The year {year} has {num_encounters} encounters after age filtering.")
        elif age == "pediatric":
            csn_df = csn_df[csn_df.age < 18]
            num_encounters = len(csn_df)
            logging.info(f"Sepy- The year {year} has {num_encounters} encounters after age filtering.")
        else:
            logging.info(f"Sepy- No specific age filter was applied")
        
        # drop duplicates
        csn_df = csn_df.drop_duplicates()
        total_num_enc = len(csn_df)
        logging.info(f"Sepy- The year {year} has {total_num_enc} encounters after filtering and dropping duplicates.")

        
        ################################################
        ############ Create Chunks of Encounters #######
        ################################################
        # calculate the chunk size based on the number of processes and the total number of encounters
        chunk_size = int(total_num_enc / num_processes)
        logging.info(f"Sepy- The ~chunk size is {chunk_size}")
        
        # split the encounters into chunks
        list_of_chunks = np.array_split(csn_df, num_processes)
        logging.info(f"Sepy- The list of chunks has {len(list_of_chunks)} unique dataframes.")
        
        # uses processor assignment number to select correct chunk
        process_list = list_of_chunks[processor_assignment].index.to_list()
        logging.info(f"Sepy- The process_list head:\n {process_list[:5]}")
        
        # create the directory for the supertables
        save_dir = SUPERTABLE_OUTPUT_PATH / str(year)
        save_dir.mkdir(exist_ok = True, parents = True)
        clinical_data_write_path = save_dir / "ClinicalData"
        clinical_data_write_path.mkdir(exist_ok = True, parents = True)
        supertable_write_path = save_dir / "Supertables"
        supertable_write_path.mkdir(exist_ok = True, parents = True)
        logging.info(f"Sepy-Directory for year {year} was set to {save_dir}")
        
        # make empty list to handle csn's with errors
        error_list = []

        ###########################################################################
        #################### Load Files for Extra Processing ######################
        ###########################################################################
        start_csn_creation = time.perf_counter()

        bounds = pd.read_csv(paths["variable_chart"])   
       
        sepyMaster_instance = make_sepyMaster(import_instance, sepyConfigs, bounds, save_dir)
        logging.info(f"Sepy- A sepyMaster instance was created for {year}")

        ###########################################################################
        ######################### Make Dicts by CSN ###############################
        ###########################################################################
        logging.info("Making supertables")
        
        # Initialize result lists
        appended_sofa_scores = []
        appended_sep3_time = []
        appended_sirs_scores = []
        appended_sep2_time = []
        appended_enc_summaries = []
        appended_comorbidity_summaries = []
        error_list = []

        # Determine batch size based on number of workers
        num_local_workers = os.cpu_count() - 1
        batch_size = num_local_workers
        logging.info(f"Sepy- Using {num_local_workers} local worker processes with batch size {batch_size}")
        
        # Process CSNs in batches
        total_csns = len(process_list)
        for batch_start in range(0, total_csns, batch_size):
            batch_end = min(batch_start + batch_size, total_csns)
            current_batch = process_list[batch_start:batch_end]

            logging.info(f"Sepy- Processing batch {batch_start//batch_size + 1} of {(total_csns + batch_size - 1)//batch_size}")
            
            # Process the batch
            batch_results = process_batch_of_csns(
                process_list=current_batch,
                sepyMaster_instance=sepyMaster_instance,
                year=year,
                start_count=batch_start
            )
            
            # Accumulate results from this batch
            for result in batch_results:
                if result['error']:
                    error_list.append(result['error'])
                    continue
                    
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
            
            # Force garbage collection after each batch
            gc.collect()
            
            # Log progress
            logging.info(f"Sepy- Completed batch {batch_start//batch_size + 1}, processed {batch_end} of {total_csns} CSNs")

        ###########################################################################
        ########################## Export Sepsis Summary ##########################
        ###########################################################################
        # create sepsis_summary directory
        base_sepsis_path = SUPERTABLE_OUTPUT_PATH / dataConfig["sepsis_summary"] / str(year)
        Path.mkdir(base_sepsis_path, exist_ok=True, parents=True)
        for subdir in dataConfig["sepsis_summary_types"]:
            Path.mkdir(base_sepsis_path / subdir, exist_ok=True, parents=True)

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
            print(appended_comorbidity_summaries)
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