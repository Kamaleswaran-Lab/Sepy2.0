# -*- coding: utf-8 -*-
"""
Elite Data Hacks
@author: Christopher S. Josef, MD
@email: csjosef@krvmail.com
"""

import sepyIMPORT as si
import pickle
import time
import glob
import sys
import numpy as np

############################## File Paths ##############################
#### data path is the parent directory for all the flat files; you'll specify each file location below
# CSJPC data_path = "C:/Users/DataSci/Desktop/em_data"
#OD data_path = "C:/Users/DataSci/OneDrive - Emory University/Sepsis Calculation/Data_Export"
# CLUSTER 
data_path = "/labs/kamaleswaranlab/MODS/Data/Emory_Data/em_data"

### grouping path is where the lists of meds, labs, & comorbs will be located
# CSJ PC groupings_path = "C:/Users/DataSci/Documents/GitHub/sepy/0.grouping"
# OD groupings_path = "C:/Users/DataSci/OneDrive - Emory University/Sepsis Calculation/groupings"
# CLUSTER 
groupings_path = "/labs/kamaleswaranlab/MODS/EliteDataHacks/sepy/new_sepy_mehak/groupings"

### Output paths is where the pickles will be written
# CSJ PC output_path = "XXXX"
# OD output_path = "C:/Users/DataSci/OneDrive - Emory University/CJ_Sepsis/5.quality_check/"
# CLUSTER 
output_path = "/labs/kamaleswaranlab/MODS/Yearly_Pickles/HolderSuperTables/"

########################################################################

############################## File Dictionaries ##############################
path_dictionary2014 = {
     # ICD9 Comorbidities
     'path_comorbid_ahrq_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_ahrq.csv')[0],
     'path_comorbid_elix_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_elix.csv')[0],
     'path_comorbid_quan_deyo_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_quan_deyo.csv')[0],
     'path_comorbid_quan_elix_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_quan_elix.csv')[0],
     'path_comorbid_ccs_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_single_ccs.csv')[0],
     # ICD10 Comorbidities
     'path_comorbid_ahrq_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_ahrq.csv')[0],
     'path_comorbid_elix_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_elix.csv')[0],
     'path_comorbid_quan_deyo_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_quan_deyo.csv')[0],
     'path_comorbid_quan_elix_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_quan_elix.csv')[0],
     'path_comorbid_ccs_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_single_ccs.csv')[0],
    
#     # Grouping Files
     'path_grouping_file_meds':glob.glob(groupings_path + '/em_all_infusion_meds*.csv')[0],   
     'path_grouping_file_labs':glob.glob(groupings_path + '/em_grouping_labs*.csv')[0],
     'path_bed_labels':glob.glob(groupings_path + '/em_bed_labels*.csv')[0],
#     # Data Files
     'path_infusion_med_file': glob.glob(data_path + '/2014/*_INFUSIONMEDS*.dsv')[0],    
     'path_lab_file': glob.glob(data_path + '/2014/*_LABS*.dsv')[0],
     'path_vitals_file': glob.glob(data_path + '/2014/*_VITALS*.dsv')[0],
     'path_vent_file': glob.glob(data_path + '/2014/*_VENT*.dsv')[0],
     'path_demographics_file': glob.glob(data_path + '/2014/*_DEMOGRAPHICS*.dsv')[0],
     'path_gcs_file': glob.glob(data_path + '/2014/*_GCS*.dsv')[0],
     'path_encounters_file': glob.glob (data_path + '/2014/*_ENCOUNTER*.dsv')[0],
     'path_cultures_file': glob.glob(data_path + '/2014/*_CULTURES*.dsv')[0],
     'path_bed_locations_file': glob.glob(data_path + '/2014/*_BEDLOCATION*.dsv')[0],
     'path_procedures_file': glob.glob(data_path + '/2014/*_ORPROCEDURES*.dsv')[0],
     'path_diagnosis_file' : glob.glob(data_path + '/2014/*_DIAGNOSIS*.dsv')[0]
     }
path_dictionary2015 = {
     # ICD9 Comorbidities
     'path_comorbid_ahrq_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_ahrq.csv')[0],
     'path_comorbid_elix_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_elix.csv')[0],
     'path_comorbid_quan_deyo_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_quan_deyo.csv')[0],
     'path_comorbid_quan_elix_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_quan_elix.csv')[0],
     'path_comorbid_ccs_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_single_ccs.csv')[0],
     # ICD10 Comorbidities
     'path_comorbid_ahrq_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_ahrq.csv')[0],
     'path_comorbid_elix_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_elix.csv')[0],
     'path_comorbid_quan_deyo_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_quan_deyo.csv')[0],
     'path_comorbid_quan_elix_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_quan_elix.csv')[0],
     'path_comorbid_ccs_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_single_ccs.csv')[0],
    
     # Grouping Files
     'path_grouping_file_meds':glob.glob(groupings_path + '/em_all_infusion_meds*.csv')[0],   
     'path_grouping_file_labs':glob.glob(groupings_path + '/em_grouping_labs*.csv')[0],
     'path_bed_labels':glob.glob(groupings_path + '/em_bed_labels*.csv')[0],
     # Data Files
     'path_infusion_med_file': glob.glob(data_path + '/2015/*_INFUSIONMEDS*.dsv')[0],    
     'path_lab_file': glob.glob(data_path + '/2015/*_LABS*.dsv')[0],
     'path_vitals_file': glob.glob(data_path + '/2015/*_VITALS*.dsv')[0],
     'path_vent_file': glob.glob(data_path + '/2015/*_VENT*.dsv')[0],
     'path_demographics_file': glob.glob(data_path + '/2015/*_DEMOGRAPHICS*.dsv')[0],
     'path_gcs_file': glob.glob(data_path + '/2015/*_GCS*.dsv')[0],
     'path_encounters_file': glob.glob (data_path + '/2015/*_ENCOUNTER*.dsv')[0],
     'path_cultures_file': glob.glob(data_path + '/2014/*_CULTURES*.dsv')[0],
     'path_bed_locations_file': glob.glob(data_path + '/2015/*_BEDLOCATION*.dsv')[0],
     'path_procedures_file': glob.glob(data_path + '/2015/*_ORPROCEDURES*.dsv')[0],
     'path_diagnosis_file' : glob.glob(data_path + '/2015/*_DIAGNOSIS*.dsv')[0]
     }
path_dictionary2016 = {
     # ICD9 Comorbidities
     'path_comorbid_ahrq_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_ahrq.csv')[0],
     'path_comorbid_elix_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_elix.csv')[0],
     'path_comorbid_quan_deyo_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_quan_deyo.csv')[0],
     'path_comorbid_quan_elix_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_quan_elix.csv')[0],
     'path_comorbid_ccs_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_single_ccs.csv')[0],
#     # ICD10 Comorbidities
     'path_comorbid_ahrq_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_ahrq.csv')[0],
     'path_comorbid_elix_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_elix.csv')[0],
     'path_comorbid_quan_deyo_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_quan_deyo.csv')[0],
     'path_comorbid_quan_elix_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_quan_elix.csv')[0],
     'path_comorbid_ccs_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_single_ccs.csv')[0],
    
#     # Grouping Files
     'path_grouping_file_meds':glob.glob(groupings_path + '/em_all_infusion_meds*.csv')[0],   
     'path_grouping_file_labs':glob.glob(groupings_path + '/em_grouping_labs*.csv')[0],
     'path_bed_labels':glob.glob(groupings_path + '/em_bed_labels*.csv')[0],
#     # Data Files
     'path_infusion_med_file': glob.glob(data_path + '/2016/*_INFUSIONMEDS*.dsv')[0],    
     'path_lab_file': glob.glob(data_path + '/2016/*_LABS*.dsv')[0],
     'path_vitals_file': glob.glob(data_path + '/2016/*_VITALS*.dsv')[0],
     'path_vent_file': glob.glob(data_path + '/2016/*_VENT*.dsv')[0],
     'path_demographics_file': glob.glob(data_path + '/2016/*_DEMOGRAPHICS*.dsv')[0],
     'path_gcs_file': glob.glob(data_path + '/2016/*_GCS*.dsv')[0],
     'path_encounters_file': glob.glob (data_path + '/2016/*_ENCOUNTER*.dsv')[0],
     'path_cultures_file': glob.glob(data_path + '/2016/*_CULTURES*.dsv')[0],
     'path_bed_locations_file': glob.glob(data_path + '/2016/*_BEDLOCATION*.dsv')[0],
     'path_procedures_file': glob.glob(data_path + '/2016/*_ORPROCEDURES*.dsv')[0],
     'path_diagnosis_file' : glob.glob(data_path + '/2016/*_DIAGNOSIS*.dsv')[0]
     }

path_dictionary2017 = {
    # ICD9 Comorbidities
    'path_comorbid_ahrq_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_ahrq.csv')[0],
    'path_comorbid_elix_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_elix.csv')[0],
    'path_comorbid_quan_deyo_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_quan_deyo.csv')[0],
    'path_comorbid_quan_elix_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_quan_elix.csv')[0],
    'path_comorbid_ccs_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_single_ccs.csv')[0],
    # ICD10 Comorbidities
    'path_comorbid_ahrq_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_ahrq.csv')[0],
    'path_comorbid_elix_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_elix.csv')[0],
    'path_comorbid_quan_deyo_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_quan_deyo.csv')[0],
    'path_comorbid_quan_elix_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_quan_elix.csv')[0],
    'path_comorbid_ccs_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_single_ccs.csv')[0],
    
    # Grouping Files
    'path_grouping_file_meds':glob.glob(groupings_path + '/em_all_infusion_meds*.csv')[0],   
    'path_grouping_file_labs':glob.glob(groupings_path + '/em_grouping_labs*.csv')[0],
    'path_bed_labels':glob.glob(groupings_path + '/em_bed_labels*.csv')[0],
    # Data Files
    'path_infusion_med_file': glob.glob(data_path + '/2017/*_INFUSIONMEDS*.dsv')[0],    
    'path_lab_file': glob.glob(data_path + '/2017/*_LABS*.dsv')[0],
    'path_vitals_file': glob.glob(data_path + '/2017/*_VITALS*.dsv')[0],
    'path_vent_file': glob.glob(data_path + '/2017/*_VENT*.dsv')[0],
    'path_demographics_file': glob.glob(data_path + '/2017/*_DEMOGRAPHICS*.dsv')[0],
    'path_gcs_file': glob.glob(data_path + '/2017/*_GCS*.dsv')[0],
    'path_encounters_file': glob.glob (data_path + '/2017/*_ENCOUNTER*.dsv')[0],
    'path_cultures_file': glob.glob(data_path + '/2017/*_CULTURES*.dsv')[0],
    'path_bed_locations_file': glob.glob(data_path + '/2017/*_BEDLOCATION*.dsv')[0],
    'path_procedures_file': glob.glob(data_path + '/2017/*_ORPROCEDURES*.dsv')[0],
    'path_diagnosis_file' : glob.glob(data_path + '/2017/*_DIAGNOSIS*.dsv')[0]
    }


path_dictionary2018 = {
     # ICD9 Comorbidities
     'path_comorbid_ahrq_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_ahrq.csv')[0],
     'path_comorbid_elix_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_elix.csv')[0],
     'path_comorbid_quan_deyo_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_quan_deyo.csv')[0],
     'path_comorbid_quan_elix_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_quan_elix.csv')[0],
     'path_comorbid_ccs_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_single_ccs.csv')[0],
     # ICD10 Comorbidities
     'path_comorbid_ahrq_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_ahrq.csv')[0],
     'path_comorbid_elix_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_elix.csv')[0],
     'path_comorbid_quan_deyo_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_quan_deyo.csv')[0],
     'path_comorbid_quan_elix_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_quan_elix.csv')[0],
     'path_comorbid_ccs_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_single_ccs.csv')[0],
    
     # Grouping Files
     'path_grouping_file_meds':glob.glob(groupings_path + '/em_all_infusion_meds*.csv')[0],   
     'path_grouping_file_labs':glob.glob(groupings_path + '/em_grouping_labs*.csv')[0],
     'path_bed_labels':glob.glob(groupings_path + '/em_bed_labels*.csv')[0],
     # Data Files
     'path_infusion_med_file': glob.glob(data_path + '/2018/*_INFUSIONMEDS*.dsv')[0],    
     'path_lab_file': glob.glob(data_path + '/2018/*_LABS*.dsv')[0],
     'path_vitals_file': glob.glob(data_path + '/2018/*_VITALS*.dsv')[0],
     'path_vent_file': glob.glob(data_path + '/2018/*_VENT*.dsv')[0],
     'path_demographics_file': glob.glob(data_path + '/2018/*_DEMOGRAPHICS*.dsv')[0],
     'path_gcs_file': glob.glob(data_path + '/2018/*_GCS*.dsv')[0],
     'path_encounters_file': glob.glob (data_path + '/2018/*_ENCOUNTER*.dsv')[0],
     'path_cultures_file': glob.glob(data_path + '/2018/*_CULTURES*.dsv')[0],
     'path_bed_locations_file': glob.glob(data_path + '/2018/*_BEDLOCATION*.dsv')[0],
     'path_procedures_file': glob.glob(data_path + '/2018/*_ORPROCEDURES*.dsv')[0],
     'path_diagnosis_file' : glob.glob(data_path + '/2018/*_DIAGNOSIS*.dsv')[0]
}
    
path_dictionary2019 = {
     # ICD9 Comorbidities
     'path_comorbid_ahrq_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_ahrq.csv')[0],
     'path_comorbid_elix_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_elix.csv')[0],
     'path_comorbid_quan_deyo_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_quan_deyo.csv')[0],
     'path_comorbid_quan_elix_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_quan_elix.csv')[0],
     'path_comorbid_ccs_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_single_ccs.csv')[0],
     # ICD10 Comorbidities
     'path_comorbid_ahrq_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_ahrq.csv')[0],
     'path_comorbid_elix_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_elix.csv')[0],
     'path_comorbid_quan_deyo_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_quan_deyo.csv')[0],
     'path_comorbid_quan_elix_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_quan_elix.csv')[0],
     'path_comorbid_ccs_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_single_ccs.csv')[0],
    
#     # Grouping Files
     'path_grouping_file_meds':glob.glob(groupings_path + '/em_all_infusion_meds*.csv')[0],   
     'path_grouping_file_labs':glob.glob(groupings_path + '/em_grouping_labs*.csv')[0],
     'path_bed_labels':glob.glob(groupings_path + '/em_bed_labels*.csv')[0],
#     # Data Files
     'path_infusion_med_file': glob.glob(data_path + '/2019/*_INFUSIONMEDS*.dsv')[0],    
     'path_lab_file': glob.glob(data_path + '/2019/*_LABS*.dsv')[0],
     'path_vitals_file': glob.glob(data_path + '/2019/*_VITALS*.dsv')[0],
     'path_vent_file': glob.glob(data_path + '/2019/*_VENT*.dsv')[0],
     'path_demographics_file': glob.glob(data_path + '/2019/*_DEMOGRAPHICS*.dsv')[0],
     'path_gcs_file': glob.glob(data_path + '/2019/*_GCS*.dsv')[0],
     'path_encounters_file': glob.glob (data_path + '/2019/*_ENCOUNTER*.dsv')[0],
     'path_cultures_file': glob.glob(data_path + '/2019/*_CULTURES*.dsv')[0],
     'path_bed_locations_file': glob.glob(data_path + '/2019/*_BEDLOCATION*.dsv')[0],
     'path_procedures_file': glob.glob(data_path + '/2019/*_ORPROCEDURES*.dsv')[0],
     'path_diagnosis_file' : glob.glob(data_path + '/2019/*_DIAGNOSIS*.dsv')[0]
}

path_dictionary2020 = {
     # ICD9 Comorbidities
     'path_comorbid_ahrq_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_ahrq.csv')[0],
     'path_comorbid_elix_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_elix.csv')[0],
     'path_comorbid_quan_deyo_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_quan_deyo.csv')[0],
     'path_comorbid_quan_elix_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_quan_elix.csv')[0],
     'path_comorbid_ccs_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_single_ccs.csv')[0],
     # ICD10 Comorbidities
     'path_comorbid_ahrq_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_ahrq.csv')[0],
     'path_comorbid_elix_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_elix.csv')[0],
     'path_comorbid_quan_deyo_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_quan_deyo.csv')[0],
     'path_comorbid_quan_elix_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_quan_elix.csv')[0],
     'path_comorbid_ccs_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_single_ccs.csv')[0],
    
     # Grouping Files
     'path_grouping_file_meds':glob.glob(groupings_path + '/em_all_infusion_meds*.csv')[0],   
     'path_grouping_file_labs':glob.glob(groupings_path + '/em_grouping_labs*.csv')[0],
     'path_bed_labels':glob.glob(groupings_path + '/em_bed_labels*.csv')[0],
     # Data Files
     'path_infusion_med_file': glob.glob(data_path + '/2020/*_INFUSIONMEDS*.dsv')[0],    
     'path_lab_file': glob.glob(data_path + '/2020/*_LABS*.dsv')[0],
     'path_vitals_file': glob.glob(data_path + '/2020/*_VITALS*.dsv')[0],
     'path_vent_file': glob.glob(data_path + '/2020/*_VENT*.dsv')[0],
     'path_demographics_file': glob.glob(data_path + '/2020/*_DEMOGRAPHICS*.dsv')[0],
     'path_gcs_file': glob.glob(data_path + '/2020/*_GCS*.dsv')[0],
     'path_encounters_file': glob.glob (data_path + '/2020/*_ENCOUNTER*.dsv')[0],
     'path_cultures_file': glob.glob(data_path + '/2020/*_CULTURES*.dsv')[0],
     'path_bed_locations_file': glob.glob(data_path + '/2020/*_BEDLOCATION*.dsv')[0],
     'path_procedures_file': glob.glob(data_path + '/2020/*_ORPROCEDURES*.dsv')[0],
     'path_diagnosis_file' : glob.glob(data_path + '/2020/*_DIAGNOSIS*.dsv')[0]
}

path_dictionary2021 = {
     # ICD9 Comorbidities
     'path_comorbid_ahrq_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_ahrq.csv')[0],
     'path_comorbid_elix_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_elix.csv')[0],
     'path_comorbid_quan_deyo_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_quan_deyo.csv')[0],
     'path_comorbid_quan_elix_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_quan_elix.csv')[0],
     'path_comorbid_ccs_ICD9':glob.glob(groupings_path + '/comorbidities/ICD9_single_ccs.csv')[0],
     # ICD10 Comorbidities
     'path_comorbid_ahrq_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_ahrq.csv')[0],
     'path_comorbid_elix_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_elix.csv')[0],
     'path_comorbid_quan_deyo_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_quan_deyo.csv')[0],
     'path_comorbid_quan_elix_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_quan_elix.csv')[0],
     'path_comorbid_ccs_ICD10':glob.glob(groupings_path + '/comorbidities/ICD10_single_ccs.csv')[0],

     # Grouping Files
     'path_grouping_file_meds':glob.glob(groupings_path + '/em_all_infusion_meds*.csv')[0],
     'path_grouping_file_labs':glob.glob(groupings_path + '/em_grouping_labs*.csv')[0],
     'path_bed_labels':glob.glob(groupings_path + '/em_bed_labels*.csv')[0],
     # Data Files
     'path_infusion_med_file': glob.glob(data_path + '/2021/*_INFUSIONMEDS*.dsv')[0],
     'path_lab_file': glob.glob(data_path + '/2021/*_LABS*.dsv')[0],
     'path_vitals_file': glob.glob(data_path + '/2021/*_VITALS*.dsv')[0],
     'path_vent_file': glob.glob(data_path + '/2021/*_VENT*.dsv')[0],
     'path_demographics_file': glob.glob(data_path + '/2021/*_DEMOGRAPHICS*.dsv')[0],
     'path_gcs_file': glob.glob(data_path + '/2021/*_GCS*.dsv')[0],
     'path_encounters_file': glob.glob (data_path + '/2021/*_ENCOUNTER*.dsv')[0],
     'path_cultures_file': glob.glob(data_path + '/2021/*_CULTURES*.dsv')[0],
     'path_bed_locations_file': glob.glob(data_path + '/2021/*_BEDLOCATION*.dsv')[0],
     'path_procedures_file': glob.glob(data_path + '/2021/*_ORPROCEDURES*.dsv')[0],
     'path_diagnosis_file' : glob.glob(data_path + '/2021/*_DIAGNOSIS*.dsv')[0]
     }


def convert_to_none_coerce_if_not(val):
    try:
        if int(val) == float(val):
            # string is int
            return np.int16(str)
        else:
            # string is numeric, but a float
            return np.nan
    except ValueError as e:
        # string cannot be parsed as a number, return nan
        return np.nan

def import_data_frames(yearly_instance):
    import_start_time = time.time()
    print('Sepy is currently reading flat files and importing them for analysis. Thank you for waiting.')
    yearly_instance.import_encounters(drop_cols=[],
                         index_col=['csn'],
                         date_cols = ['ed_presentation_time',
                                      'hospital_admission_date_time',
                                      'hospital_discharge_date_time'])
    
    yearly_instance.import_demographics(drop_cols=[],
                                        index_col = ['pat_id'],
                                        date_cols=['dob'])
                
    yearly_instance.import_infusion_meds(drop_cols=[],
                             numeric_cols = ['order_med_id','medication_id'],
                             anti_infective_group_name = 'anti-infective',
                             vasopressor_group_name = 'vasopressor',
                             index_col = ['csn'],
                             date_cols = ['med_order_time',
                                          'med_action_time',
                                          'med_start','med_stop'])
    yearly_instance.import_labs(drop_cols=[],
                                group_cols =['physionet','import','super_table_col_name', 'component_id'],
                                date_cols = ['collection_time','lab_result_time'],
                                index_col = ['csn'],
                                numeric_cols = None)
    
    yearly_instance.import_vitals(drop_cols = [],
                                  numeric_cols = yearly_instance.vital_col_names,
                                  index_col = ['csn'],
                                  date_cols = ['recorded_time'],
                                  merge_cols = None)
    yearly_instance.import_vent(drop_cols = [],
                                index_col = ['csn'],
                                numeric_cols = ['vent_rate_set','vent_tidal_rate_set','vent_tidal_rate_exhaled', 'peep','fio2'],
                                date_cols = ['recorded_time','vent_start_time','vent_stop_time'])
    
    yearly_instance.import_gcs(drop_cols = [],
                               index_col = ['csn'],
                               numeric_col =['gcs_eye_score','gcs_verbal_score','gcs_motor_score', 'gcs_total_score'],
                               date_cols = ['recorded_time'])

    yearly_instance.import_cultures(drop_cols = [],
                                    index_col = ['csn'],
                                    date_cols = ['specimen_collect_time','order_time','lab_result_time'])

    yearly_instance.import_bed_locations(drop_cols = [],
                                         index_col = ['csn'],
                                         date_cols = ['bed_location_start','bed_location_end'])
    
    yearly_instance.import_procedures(drop_cols = [],
                                      index_col = ['csn'],
                                      date_cols = ['surgery_date','in_or_dttm','procedure_start_dttm','procedure_comp_dttm','out_or_dttm'])

    yearly_instance.import_diagnosis(drop_cols = [],
                                     index_col = ['csn'],
                                     date_cols = ['dx_time_date'])
    print(f"Sepy took {time.time()-import_start_time} (s) to create a yearly pickle.")


if __name__ == '__main__':

    try:
        #starts yearly pickle timmer
        start = time.perf_counter()
        
        #accepts command line argument for year
        year = int(sys.argv[1])
        
        pickle_file_name = output_path + 'em_y' + str(year) + '.pickle'
        print(pickle_file_name)
        
        path_dictionary = "path_dictionary" +  str(year)
        print(f'File locations were taken from the path dictionary: {path_dictionary}')
        
        yearly_instance = si.sepyIMPORT(eval(path_dictionary),"|")
        print(f'An instance of the sepyIMPORT class was created for {year}')
        import_data_frames(yearly_instance)
    
        with open(pickle_file_name, 'wb') as handle:
            pickle.dump(yearly_instance, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f'Time to create {year}s data and write to pickles was {time.perf_counter()-start} (s)')
    
    except Exception as e: 
         print(e)
         print (f'There was an error with the class instantiation for {year}')
