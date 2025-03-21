# -*- coding: utf-8 -*-
"""
This module dumps pickels for each encounter in a given year and creates dictionaries indexed by CSN.
The dictionaries are used to create summary reports for each encounter.

Author: mehak
Created on Wed Mar 29 23:33:04 2023
Version: 0.1

Author: Jack F. Regan
Edited: 2025-03-01
Version: 0.2
Changes:
- updates to functiondocumentation
- updates to proper variable naming
- removed ICD9 comorbidity dictionaries
- corrected syntax error in directory creation, line 216

"""
import sys
import pickle
import time

from pathlib import Path
import pandas as pd
import numpy as np
import sepyDICT as sd

BED_UNIT_CSV_FNAME = Path(
    "/labs/kamaleswaranlab/MODS/EliteDataHacks/sepy/new_sepy_mehak/bed_units_to_icu_AH.csv"
)
VARIABLE_BOUNDS_CSV_FNAME = Path(
    "/labs/kamaleswaranlab/MODS/EliteDataHacks/sepy/new_sepy_mehak/Variable_Chart.xlsx"
)
DIALYSIS_INFO_CSN_FNAME = Path(
    "/labs/kamaleswaranlab/MODS/EliteDataHacks/sepy/new_sepy_mehak/PEACH_HD_CRRT(1).csv"
)


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
    Summarizes the comorbidity data for a single patient encounter by extracting ICD-10-related comorbidity counts
    and stores them in global dictionaries.

    Args:
        csn (int or str): The unique encounter ID (CSN) for the patient encounter.
        instance (sepyDICT): An instance of the sepyDICT class, containing comorbidity-related data.
    """
    # ICD10
    # ahrq_ICD10_dict[csn]=instance.ahrq_ICD10_PerCSN.icd_count
    # elix_ICD10_dict[csn]=instance.elix_ICD10_PerCSN.icd_count
    quan_deyo_ICD10_dict[encounter_csn] = encounter_instance.quan_deyo_ICD10_PerCSN.icd_count
    quan_elix_ICD10_dict[encounter_csn] = encounter_instance.quan_elix_ICD10_PerCSN.icd_count
    # ccs_ICD10_dict[csn]=instance.ccs_ICD10_PerCSN.icd_count


############################## Initialize Empty Summaries ##############################
# ICD10
# ahrq_ICD10_dict={}
# elix_ICD10_dict={}
quan_deyo_ICD10_dict = {}
quan_elix_ICD10_dict = {}
# ccs_ICD10_dict={}

# other summaries
appended_sofa_scores = []
appended_sep3_time = []
appended_sirs_scores = []
appended_sep2_time = []
appended_enc_summaries = []

start = time.perf_counter()

if __name__ == "__main__":
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
        print(sys.version)
        csn_list_file_name = Path(sys.argv[1])
        # imports file location for CSN list
        print(f"MkDct- The CSN List location: {csn_list_file_name}")
        # sets directoy for yearly pickles
        pickle_path = Path(sys.argv[2])
        print(f"MkDct- The pickle directory: {pickle_path}")
        # sets directory for output
        output_path = Path(sys.argv[3])
        print(f"MkDct- The output directory: {output_path}")
        # total number of tasks/processes
        num_processes = int(sys.argv[4])
        print(f"MkDct- The tot num of processes: {num_processes}")
        # task number
        processor_assignment = int(sys.argv[5])
        print(f"MkDct- This task is: {processor_assignment}")
        # year num provided in bash
        bash_year = int(sys.argv[6])
        print(f"MkDct- The import year is: {bash_year}")
    except (IndexError, ValueError, TypeError) as e:
        print(
            f"MkDct- There was an error importing one of the arguments: {type(e).__name__}."
        )
        print(f"MkDct- You are trying to load the following CSN list {sys.argv[1]}")
        print(f"MkDct- You are trying to use this many processors {sys.argv[2]}")

    ########### Create Encounter List Based on Processor Assignment ###########
    # Cohort selector
    ed = 0
    in_pt = 1
    icu = 0
    adult = 1
    vent_row = 0
    vent_start = 0
    # reads the list of csns
    csn_df = pd.read_csv(csn_list_file_name, sep="|", header=0)
    # only keep csns that meet specified year
    csn_df = csn_df[(csn_df.in_pt == in_pt) & (csn_df.adult == adult)]
    print(f"MkDct- A total of {csn_df.shape[0]} encounters were selected")
    csn_df = csn_df[["csn", "year"]]
    # drop duplicates and filter out only selected years
    csn_df = csn_df.drop_duplicates()
    csn_df = csn_df[csn_df.year == bash_year]
    total_num_enc = len(csn_df)
    print(f"MkDct- The year {bash_year} has {total_num_enc} encounters.")
    # breaks encounter list into chunks, selects correct chunk based on process num
    chunk_size = int(total_num_enc / num_processes)
    print(f"MkDct- The ~chunk size is {chunk_size}")
    # split list
    list_of_chunks = np.array_split(csn_df, num_processes)
    print(f"MkDct- The list of chunks has {len(list_of_chunks)} unique dataframes.")
    # uses processor assignment to select correct chunk
    process_list = list_of_chunks[processor_assignment]["csn"]
    print(f"MkDct- The process_list head:\n {process_list.head()}")
    # select correct pickle by year
    pickle_name = pickle_path / ("em_y" + str(bash_year) + ".pickle")
    print(f"MkDct- The following pickle is being read: {pickle_name}")
    try:
        # reads the IMPORT class instance (i.e.  1 year of patient data)
        pickle_load_time = time.perf_counter()
        yearly_instance = pd.read_pickle(pickle_name)
        print(
            f"MkDct-Pickle from year {bash_year} was loaded in {time.perf_counter()-pickle_load_time}s."
        )
        print("-----------LOADED YEARLY PICKLE FILE!!!!---------------")
        # if success, make a dir for this year's encounters
        pickle_write_path = output_path / str(bash_year)
        pickle_write_path.mkdir(exist_ok=True)
        print(f"MkDct-Directory for year {bash_year} was set to {pickle_write_path}")
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
            print("MkDct- The bed to unit mapping file is not formatted correctly.")
        bounds = pd.read_excel(VARIABLE_BOUNDS_CSV_FNAME, engine="openpyxl")
        dialysis = pd.read_csv(DIALYSIS_INFO_CSN_FNAME)
        dialysis_year = dialysis.loc[
            dialysis["Encounter Number"].isin(csn_df["csn"].values)
        ]
        ################################ Make Dicts by CSN ################################
        for count, csn in enumerate(process_list, start=1):
            try:
                print(f"MkDct- Processing patient csn: {csn}, {count} of {chunk_size} for year {bash_year}")
                instance = process_csn(csn, pickle_write_path, bed_to_unit_mapping, bounds, dialysis_year, yearly_instance)
                print("MkDct- Instance created")
                # Running summaries with error handling
                try:
                    sofa_summary(csn, instance)
                except Exception as e:
                    print(f"MkDct- Error in Sofa Summary for csn {csn}: {e}")
                try:
                    sepsis3_summary(csn, instance)
                except Exception as e:
                    print(f"MkDct- Error in Sepsis 3 Summary for csn {csn}: {e}")
                try:
                    sirs_summary(csn, instance)
                except Exception as e:
                    print(f"MkDct- Error in SIRS Summary for csn {csn}: {e}")
                try:
                    sepsis2_summary(csn, instance)
                except Exception as e:
                    print(f"MkDct- Error in Sepsis 2 Summary for csn {csn}: {e}")
                try:
                    enc_summary(instance)
                except Exception as e:
                    print(f"MkDct- Error in Encounter Summary for csn {csn}: {e}")
                try:
                    comorbidity_summary(csn, instance)
                except Exception as e:
                    print(f"MkDct- Error in Comorbidity Summary for csn {csn}: {e}")
                print(f"MkDct- Encounter {count} of {chunk_size} is complete!")
            except Exception as e:
                print(f"MkDct- Error processing csn {csn}: {e}")
                error_list.append([csn, e.args[0]])
                print(f"MkDct- The following csn had an error: {csn}")
        ############################# Export Sepsis Summary #############################
        # create sepsis_summary directory
        Path.mkdir(output_path / "em_sepsis_summary", exist_ok=True)
        Path.mkdir(output_path / "em_sepsis_summary" / "sofa_summary", exist_ok=True)
        Path.mkdir(output_path / "em_sepsis_summary" / "sep3_summary", exist_ok=True)
        Path.mkdir(output_path / "em_sepsis_summary" / "sirs_summary", exist_ok=True)
        Path.mkdir(output_path / "em_sepsis_summary" / "sep2_summary", exist_ok=True)
        Path.mkdir(
            output_path / "em_sepsis_summary" / "encounter_summary", exist_ok=True
        )
        Path.mkdir(output_path / "em_sepsis_summary" / "error_summary", exist_ok=True)

        # ICD10 co-morbid
        # Path.mkdir(output_path / 'em_sepsis_summary' / 'ahrq_ICD10_summary', exist_ok = True)
        # Path.mkdir(output_path / 'em_sepsis_summary' / 'elix_ICD10_summary', exist_ok = True)
        Path.mkdir(
            output_path / "em_sepsis_summary" / "quan_deyo_ICD10_summary", exist_ok=True
        )
        Path.mkdir(
            output_path / "em_sepsis_summary" / "quan_elix_ICD10_summary", exist_ok=True
        )
        # Path.mkdir(output_path / 'em_sepsis_summary' / 'ccs_ICD10_summary', exist_ok = True)
        # write general files
        # Save encounter summary
        UNIQUE_FILE_ID = f"{processor_assignment}_{bash_year}"
        base_path = output_path / "em_sepsis_summary"
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
        pd.DataFrame.from_dict(quan_deyo_ICD10_dict).T.to_csv(
            base_path
            / "quan_deyo_ICD10_summary"
            / ("quan_deyo_ICD10_summary_" + UNIQUE_FILE_ID + ".csv"),
            index=True,
            index_label="csn",
        )
        pd.DataFrame.from_dict(quan_elix_ICD10_dict).T.to_csv(
            base_path
            / "quan_elix_ICD10_summary"
            / ("quan_elix_ICD10_summary_" + UNIQUE_FILE_ID + ".csv"),
            index=True,
            index_label="csn",
        )
        # pd.DataFrame.from_dict(ccs_ICD10_dict).T.to_csv(output_path / 'em_sepsis_summary' / 'ccs_ICD10_summary' / ('ccs_ICD10_summary_'+ unique_file_id +'.csv'), index = True, index_label='csn')
        print(
            f"MkDct- Time to create write encounter pickles for {bash_year} was {time.perf_counter() - start_csn_creation} (s)"
        )
    except Exception as e:
        print(f"MkDct- Could not find or open the pickle for year {bash_year}: {e}")
