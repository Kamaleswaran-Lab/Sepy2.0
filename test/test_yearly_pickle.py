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

import sys
sys.path.append(os.path.expandvars("$HOME/Sepy2.0/"))

import sepyIMPORT as si
import sepyDICT as sd
import utils

import importlib
importlib.reload(si)
importlib.reload(sd)


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
        method = getattr(yearly_instance, method_name, None)
        if callable(method):
            # check if method requires numeric_cols parameter and access list in sepyIMPORT instnace
            if "numeric_cols" in params and isinstance(params["numeric_cols"], str):
                params["numeric_cols"] = getattr(yearly_instance, params["numeric_cols"], None)
        method(**params)
    logging.info(f"Sepy took {time.time() - import_start_time} (s) to create a yearly pickle.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=str, required=True)
    args = parser.parse_args()
    year = args.year

    dataConfig_path = '../configurations/emory_config_oddjobs.yaml'
    sepyIMPORTConfigs_path = '../configurations/dict_config.yaml'
    sepyDICTConfigs_path = '../configurations/dict_config.yaml'
    num_processes = 8
    processor_assignment = 0
    dataConfig = utils.load_yaml(dataConfig_path)
    sepyIMPORTConfigs = utils.load_yaml(sepyIMPORTConfigs_path)
    sepyDICTConfigs = utils.load_yaml(sepyDICTConfigs_path)

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
                f"{DATA_PATH}/*{combined_file[1]}*"
            
            )[0]
        except IndexError:
            logging.error(f"Sepy- could not find combined file for {combined_file[1]}")
    print(paths)
    file_dictionary = paths


    import_instance = si.sepyIMPORT(paths, sepyIMPORTConfigs, dataConfig["yearly_instance"], create_dataframes = True, save_dataframes = True, save_path = YEARLY_DICTIONARY_FILE_NAME)

if __name__ == "__main__":
    main()