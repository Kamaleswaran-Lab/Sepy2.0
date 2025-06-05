import yaml
import pandas as pd
import logging


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

