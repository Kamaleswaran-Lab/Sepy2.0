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
import dask.dataframe as dd
import pandas as pd

############################## File Paths ##############################
#### data path is the parent directory for all the flat files; you'll specify each file location below
# CSJPC data_path = "M:/BoxSync/Grady Decomp & TDAP/Grady Data"
# OD data_path = XXX
# CLUSTER 
data_path = "/labs/kamaleswaranlab/MODS/Data/Grady_Data"

### grouping path is where the lists of meds, labs, & comorbs will be located
# CSJ PC groupings_path = "C:/Users/DataSci/Documents/GitHub/sepy/0.grouping"
# OD groupings_path = "C:/Users/DataSci/Documents/GitHub/sepy/0.grouping"
# CLUSTER 
groupings_path = "/labs/kamaleswaranlab/MODS/EliteDataHacks/sepy/0.grouping"

### Output paths is where the pickles will be written
# CSJ PC output_path = "XXXX"
# OD  output_path = "C:/Users/DataSci/OneDrive - Emory University/CJ_Sepsis/5.quality_check/"
# CLUSTER 
output_path = "/labs/kamaleswaranlab/MODS/Yearly_Pickles/"

########################################################################

############################## File Dictionaries ##############################
# A dictionary is created for each year that has the exact file location
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

    # Grouping Files
    'path_grouping_file_meds':glob.glob(groupings_path + '/gr_all_infusion_meds*.csv')[0],   
    'path_grouping_file_labs':glob.glob(groupings_path + '/gr_grouping_labs*.csv')[0],
    'path_bed_labels':glob.glob(groupings_path + '/gr_bed_labels*.csv')[0],
    # Data Files
    'path_infusion_med_file': glob.glob(data_path + '/2. Fluids & Meds/Infusion Medications/infusion_meds_' + '2014_*.txt')[0],    
    'path_lab_file': glob.glob(data_path + '/3. Labs & Cultures/Labs/' + 'lab_2014*.txt')[0],
    'path_vitals_file': glob.glob(data_path + '/4. Patient Assessments/Vitals/vitals_2014*.txt')[0],
    'path_vent_file': glob.glob(data_path + '/6. Vent/vent_data_2014*.txt')[0],
    'path_demographics_file': glob.glob(data_path + '/1. Administrative Attributes/Demographics/demographics_2014*.txt')[0],
    'path_gcs_file': glob.glob(data_path + '/4. Patient Assessments/GCS/gcs_2014*.txt')[0],
    'path_encounters_file': glob.glob (data_path + '/1. Administrative Attributes/Encounters/encounter_2014*.txt')[0],
    'path_cultures_file': glob.glob(data_path + '/3. Labs & Cultures/Cultures/cultures_2014*.txt')[0],
    'path_bed_locations_file': glob.glob(data_path + '/1. Administrative Attributes/Bed Locations/bed_location_2014*.txt')[0],
    'path_procedures_file': glob.glob(data_path + '/5. ICD Codes/OR Procedures/or_procedures_2014*.txt')[0],
    'path_diagnosis_file' : glob.glob(data_path + '/5. ICD Codes/Diagnosis/diagnoses_2014*.txt')[0]
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
    'path_grouping_file_meds':glob.glob(groupings_path + '/gr_all_infusion_meds*.csv')[0],   
    'path_grouping_file_labs':glob.glob(groupings_path + '/gr_grouping_labs*.csv')[0],
    'path_bed_labels':glob.glob(groupings_path + '/gr_bed_labels*.csv')[0],
    # Data Files   
    'path_infusion_med_file': glob.glob(data_path + '/2. Fluids & Meds/Infusion Medications/infusion_meds_' + '2015_*.txt')[0],
    'path_lab_file': glob.glob(data_path + '/3. Labs & Cultures/Labs/lab_2015*.txt')[0], 
    'path_vitals_file': glob.glob(data_path + '/4. Patient Assessments/Vitals/vitals_2015*.txt')[0],
    'path_vent_file': glob.glob(data_path + '/6. Vent/vent_data_2015*.txt')[0],
    'path_demographics_file': glob.glob(data_path + '/1. Administrative Attributes/Demographics/demographics_2015*.txt')[0],
    'path_gcs_file': glob.glob(data_path + '/4. Patient Assessments/GCS/gcs_2015*.txt')[0],
    'path_encounters_file': glob.glob(data_path + '/1. Administrative Attributes/Encounters/encounter_2015*.txt')[0],
    'path_cultures_file': glob.glob(data_path + '/3. Labs & Cultures/Cultures/cultures_2015*.txt')[0],
    'path_bed_locations_file': glob.glob(data_path + '/1. Administrative Attributes/Bed Locations/bed_location_2015*.txt')[0],
    'path_procedures_file': glob.glob(data_path + '/5. ICD Codes/OR Procedures/or_procedures_2015*.txt')[0],
    'path_diagnosis_file' : glob.glob(data_path + '/5. ICD Codes/Diagnosis/diagnoses_2015*.txt')[0]
    }

path_dictionary2016 = {
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
    'path_grouping_file_meds':glob.glob(groupings_path + '/gr_all_infusion_meds*.csv')[0],   
    'path_grouping_file_labs':glob.glob(groupings_path + '/gr_grouping_labs*.csv')[0],
    'path_bed_labels':glob.glob(groupings_path + '/gr_bed_labels*.csv')[0],
    # Data Files
    'path_infusion_med_file': glob.glob(data_path + '/2. Fluids & Meds/Infusion Medications/infusion_meds_2016_*.txt')[0],
    'path_lab_file': glob.glob(data_path + '/3. Labs & Cultures/Labs/' + 'lab_2016*.txt')[0],
    'path_vitals_file': glob.glob(data_path + '/4. Patient Assessments/Vitals/vitals_2016*.txt')[0],
    'path_vent_file': glob.glob(data_path + '/6. Vent/vent_data_2016*.txt')[0],
    'path_demographics_file': glob.glob(data_path + '/1. Administrative Attributes/Demographics/demographics_2016*.txt')[0],
    'path_gcs_file': glob.glob(data_path + '/4. Patient Assessments/GCS/gcs_2016*.txt')[0],
    'path_encounters_file': glob.glob(data_path + '/1. Administrative Attributes/Encounters/encounter_2016*.txt')[0],
    'path_cultures_file': glob.glob(data_path + '/3. Labs & Cultures/Cultures/cultures_2016*.txt')[0],
    'path_bed_locations_file': glob.glob(data_path + '/1. Administrative Attributes/Bed Locations/bed_location_2016*.txt')[0],
    'path_procedures_file': glob.glob(data_path + '/5. ICD Codes/OR Procedures/or_procedures_2016*.txt')[0],
    'path_diagnosis_file' : glob.glob(data_path + '/5. ICD Codes/Diagnosis/diagnoses_2016*.txt')[0]
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
    'path_grouping_file_meds':glob.glob(groupings_path + '/gr_all_infusion_meds*.csv')[0],   
    'path_grouping_file_labs':glob.glob(groupings_path + '/gr_grouping_labs*.csv')[0],
    'path_bed_labels':glob.glob(groupings_path + '/gr_bed_labels*.csv')[0],
    # Data Files
    'path_infusion_med_file': glob.glob(data_path + '/2. Fluids & Meds/Infusion Medications/infusion_meds_' + '2017*.txt')[0],
    'path_lab_file': glob.glob(data_path + '/3. Labs & Cultures/Labs/' + 'lab_2017*.txt')[0],
    'path_vitals_file': glob.glob(data_path + '/4. Patient Assessments/Vitals/vitals_2017*.txt')[0],
    'path_vent_file': glob.glob(data_path + '/6. Vent/vent_data_2017*.txt')[0],
    'path_demographics_file': glob.glob(data_path + '/1. Administrative Attributes/Demographics/demographics_2017*.txt')[0],
    'path_gcs_file': glob.glob(data_path + '/4. Patient Assessments/GCS/gcs_2017*.txt')[0],
    'path_encounters_file': glob.glob(data_path + '/1. Administrative Attributes/Encounters/encounter_2017*.txt')[0],
    'path_cultures_file': glob.glob(data_path + '/3. Labs & Cultures/Cultures/cultures_2017*.txt')[0],
    'path_bed_locations_file': glob.glob(data_path + '/1. Administrative Attributes/Bed Locations/bed_location_2017*.txt')[0],
    'path_procedures_file': glob.glob(data_path + '/5. ICD Codes/OR Procedures/or_procedures_2017*.txt')[0],
    'path_diagnosis_file' : glob.glob(data_path + '/5. ICD Codes/Diagnosis/diagnoses_2017*.txt')[0]
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
    'path_grouping_file_meds':glob.glob(groupings_path + '/gr_all_infusion_meds*.csv')[0],   
    'path_grouping_file_labs':glob.glob(groupings_path + '/gr_grouping_labs*.csv')[0],
    'path_bed_labels':glob.glob(groupings_path + '/gr_bed_labels*.csv')[0],
    # Data Files
    'path_infusion_med_file': glob.glob(data_path + '/2. Fluids & Meds/Infusion Medications/infusion_meds_2018_*.txt')[0],
    'path_lab_file': glob.glob(data_path + '/3. Labs & Cultures/Labs/lab_2018*.txt')[0],
    'path_vitals_file': glob.glob(data_path + '/4. Patient Assessments/Vitals/vitals_2018*.txt')[0],
    'path_vent_file': glob.glob(data_path + '/6. Vent/vent_data_2018*.txt')[0],
    'path_demographics_file': glob.glob(data_path + '/1. Administrative Attributes/Demographics/demographics_2018*.txt')[0],
    'path_gcs_file': glob.glob(data_path + '/4. Patient Assessments/GCS/gcs_2018*.txt')[0],
    'path_encounters_file': glob.glob(data_path + '/1. Administrative Attributes/Encounters/encounter_2018*.txt')[0],
    'path_cultures_file': glob.glob(data_path + '/3. Labs & Cultures/Cultures/cultures_2018*.txt')[0],
    'path_bed_locations_file': glob.glob(data_path + '/1. Administrative Attributes/Bed Locations/bed_location_2018*.txt')[0],
    'path_procedures_file': glob.glob(data_path + '/5. ICD Codes/OR Procedures/or_procedures_2018*.txt')[0],
    'path_diagnosis_file' : glob.glob(data_path + '/5. ICD Codes/Diagnosis/diagnoses_2018*.txt')[0]
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
    
    # Grouping Files
    'path_grouping_file_meds':glob.glob(groupings_path + '/gr_all_infusion_meds*.csv')[0],   
    'path_grouping_file_labs':glob.glob(groupings_path + '/gr_grouping_labs*.csv')[0],
    'path_bed_labels':glob.glob(groupings_path + '/gr_bed_labels*.csv')[0],
    # Data Files
    'path_infusion_med_file':glob.glob(data_path + '/2. Fluids & Meds/Infusion Medications/infusion_meds_2019_*.txt')[0],   
    'path_lab_file': glob.glob(data_path + '/3. Labs & Cultures/Labs/lab_2019*.txt')[0],   
    'path_vitals_file': glob.glob(data_path + '/4. Patient Assessments/Vitals/vitals_2019*.txt')[0],
    'path_vent_file': glob.glob(data_path + '/6. Vent/vent_data_2019*.txt')[0],
    'path_demographics_file': glob.glob(data_path + '/1. Administrative Attributes/Demographics/demographics_2019*.txt')[0],
    'path_gcs_file': glob.glob(data_path + '/4. Patient Assessments/GCS/gcs_2019*.txt')[0],
    'path_encounters_file': glob.glob(data_path + '/1. Administrative Attributes/Encounters/encounters_2019*.txt')[0],
    'path_cultures_file': glob.glob(data_path + '/3. Labs & Cultures/Cultures/cultures_2019*.txt')[0],
    'path_bed_locations_file': glob.glob(data_path + '/1. Administrative Attributes/Bed Locations/bed_location_2019*.txt')[0],
    'path_procedures_file': glob.glob(data_path + '/5. ICD Codes/OR Procedures/or_procedures_2019*.txt')[0],
    'path_diagnosis_file' : glob.glob(data_path + '/5. ICD Codes/Diagnosis/diagnoses_2019*.txt')[0]
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
    'path_grouping_file_meds':glob.glob(groupings_path + '/gr_all_infusion_meds*.csv')[0],   
    'path_grouping_file_labs':glob.glob(groupings_path + '/gr_grouping_labs*.csv')[0],
    'path_bed_labels':glob.glob(groupings_path + '/gr_bed_labels*.csv')[0],
    # Data Files
    'path_infusion_med_file': glob.glob(data_path + '/2. Fluids & Meds/Infusion Medications/infusion_meds_2020_*.txt')[0],
    'path_lab_file': glob.glob(data_path + '/3. Labs & Cultures/Labs/lab_2020*.txt')[0],  
    'path_vitals_file': glob.glob(data_path + '/4. Patient Assessments/Vitals/vitals_2020*.txt')[0],
    'path_vent_file': glob.glob(data_path + '/6. Vent/vent_data_2020*.txt')[0],
    'path_demographics_file': glob.glob(data_path + '/1. Administrative Attributes/Demographics/demographics_2020*.txt')[0],
    'path_gcs_file':glob.glob(data_path + '/4. Patient Assessments/GCS/gcs_2020*.txt')[0],
    'path_encounters_file': glob.glob(data_path + '/1. Administrative Attributes/Encounters/encounter_2020*.txt')[0],
    'path_cultures_file': glob.glob(data_path + '/3. Labs & Cultures/Cultures/cultures_2020*.txt')[0],
    'path_bed_locations_file': glob.glob(data_path + '/1. Administrative Attributes/Bed Locations/bed_location_2020*.txt')[0],
    'path_procedures_file': glob.glob(data_path + '/5. ICD Codes/OR Procedures/or_procedures_2020*.txt')[0],
    'path_diagnosis_file' : glob.glob(data_path + '/5. ICD Codes/Diagnosis/diagnoses_2020*.txt')[0]
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
    'path_grouping_file_meds':glob.glob(groupings_path + '/gr_all_infusion_meds*.csv')[0],   
    'path_grouping_file_labs':glob.glob(groupings_path + '/gr_grouping_labs*.csv')[0],
    'path_bed_labels':glob.glob(groupings_path + '/gr_bed_labels*.csv')[0],
    # Data Files
    'path_infusion_med_file': glob.glob(data_path + '/2. Fluids & Meds/Infusion Medications/infusion_meds_2021_09062023.txt')[0],
    'path_lab_file': glob.glob(data_path + '/3. Labs & Cultures/Labs/labs_2021_09062023.txt')[0],  
    'path_vitals_file': glob.glob(data_path + '/4. Patient Assessments/Vitals/vitals_2021_09062023.txt')[0],
    'path_vent_file': glob.glob(data_path + '/6. Vent/vent_2021_09062023.txt')[0],
    'path_demographics_file': glob.glob(data_path + '/1. Administrative Attributes/Demographics/demographics_2014-2022*.txt')[0],
    'path_gcs_file':glob.glob(data_path + '/4. Patient Assessments/GCS/gcs_2014-2022*.txt')[0],
    'path_encounters_file': glob.glob(data_path + '/1. Administrative Attributes/Encounters/encounter_2021_09062023.txt')[0],
    'path_cultures_file': glob.glob(data_path + '/3. Labs & Cultures/Cultures/cultures_2021_09062023.txt')[0],
    'path_bed_locations_file': glob.glob(data_path + '/1. Administrative Attributes/Bed Locations/bed_location_2021_09062023.txt')[0],
    'path_procedures_file': glob.glob(data_path + '/5. ICD Codes/OR Procedures/or_2021_09062023.txt')[0],
    'path_diagnosis_file' : glob.glob(data_path + '/5. ICD Codes/Diagnosis/diagnosis_2021_09062023.txt')[0]
    }


path_dictionary2022 = {
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
    'path_grouping_file_meds':glob.glob(groupings_path + '/gr_all_infusion_meds*.csv')[0],   
    'path_grouping_file_labs':glob.glob(groupings_path + '/gr_grouping_labs*.csv')[0],
    'path_bed_labels':glob.glob(groupings_path + '/gr_bed_labels*.csv')[0],
    # Data Files
    'path_infusion_med_file': glob.glob(data_path + '/2. Fluids & Meds/Infusion Medications/infusion_meds_2022_*.txt')[0],
    'path_lab_file': glob.glob(data_path + '/3. Labs & Cultures/Labs/lab_2022*.txt')[0],  
    'path_vitals_file': glob.glob(data_path + '/4. Patient Assessments/Vitals/vitals_2022*.txt')[0],
    'path_vent_file': glob.glob(data_path + '/6. Vent/vent_2022*.txt')[0],
    'path_demographics_file': glob.glob(data_path + '/1. Administrative Attributes/Demographics/demographics_2022*.txt')[0],
    'path_gcs_file':glob.glob(data_path + '/4. Patient Assessments/GCS/gcs_2022*.txt')[0],
    'path_encounters_file': glob.glob(data_path + '/1. Administrative Attributes/Encounters/encounter_2022*.txt')[0],
    'path_cultures_file': glob.glob(data_path + '/3. Labs & Cultures/Cultures/cultures_2022*.txt')[0],
    'path_bed_locations_file': glob.glob(data_path + '/1. Administrative Attributes/Bed Locations/2021-2022 Decomp Files Rishi/bed_location_2022*.txt')[0],
    'path_procedures_file': glob.glob(data_path + '/5. ICD Codes/OR Procedures/or_procedures_2022*.txt')[0],
    'path_diagnosis_file' : glob.glob(data_path + '/5. ICD Codes/Diagnosis/diagnoses_2022*.txt')[0]
    }


def convert_to_none_coerce_if_not(val: str):
    try:
        if int(str) == float(str):
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
    
    yearly_instance.import_encounters(drop_cols = ['study_id','har','mrn'],
                                      index_col = ['csn'],
                                      date_cols = ['ed_presentation_time',
                                                   'hospital_admission_date_time',
                                                   'hospital_discharge_date_time'])

    yearly_instance.import_demographics(drop_cols = ['study_id'],
                                        index_col = ['pat_id'],
                                        date_cols = ['dob'])
                
    yearly_instance.import_infusion_meds(drop_cols = ['study_id','har','mrn'],
                                         numeric_cols = [],
                                         anti_infective_group_name = 'anti-infective',
                                         vasopressor_group_name = 'vasopressor',
                                         #na_values = ['NULL'],
                                         index_col = ['csn'],
                                         date_cols = ['med_order_time',
                                                      'med_action_time',
                                                      'med_start','med_stop'])
    
    #yearly_instance.import_labs(drop_cols = ['study_id','har','mrn'],
    #                            group_cols = ['physionet','import','super_table_col_name', 'component_id'],
    #                            date_cols = ['collection_time','lab_result_time'],
    #                            index_col = ['csn'],
    #                            numeric_cols = None)
    yearly_instance.df_labs = pd.read_pickle('/labs/kamaleswaranlab/MODS/Encounter_Pickles/gr/grady_2022_labs_reindexed.pkl')
    yearly_instance.import_vitals(drop_cols = ['study_id','har','mrn'],
                                  numeric_cols = yearly_instance.vital_col_names,
                                  index_col = ['csn'],
                                  date_cols = ['recorded_time'],
                                  merge_cols = [['end_tidal_co2_1','end_tidal_co2_2' ,'end_tidal_co2']])
    
    yearly_instance.import_vent(drop_cols = ['study_id','har','mrn'],
                                index_col = ['csn'],
                                numeric_cols = ['vent_rate_set','vent_tidal_rate_set','vent_tidal_rate_exhaled', 'peep','fio2'],
                                date_cols = ['vent_recorded_time','vent_start_time','vent_stop_time', 'recorded_time'])
    
    yearly_instance.import_gcs(drop_cols = ['study_id','har','mrn'],
                               index_col = ['csn'],
                               numeric_col = ['gcs_eye_score','gcs_verbal_score','gcs_motor_score', 'gcs_total_score'],
                               date_cols = ['recorded_time'])
    
    yearly_instance.import_cultures(drop_cols = ['study_id','har','mrn'],
                                    index_col = ['csn'],
                                    date_cols = ['specimen_collect_time','order_time','lab_result_time'])
    
    yearly_instance.import_bed_locations(drop_cols = ['study_id','har','mrn'],
                                         index_col = ['csn'],
                                         date_cols = ['bed_location_start','bed_location_end'])
    
    yearly_instance.import_procedures(drop_cols = ['study_id','har','mrn'],
                                      index_col = ['csn'],
                                      date_cols = ['surgery_date','in_or_dttm','procedure_start_dttm','procedure_comp_dttm','out_or_dttm'])

    yearly_instance.import_diagnosis(drop_cols = ['study_id','har','mrn'],
                                     index_col = ['csn'],
                                     date_cols = ['dx_time_date'])
    
    print(f'Sepy took {time.time()-import_start_time} (s) to create a yearly pickle.')

########################################################################
if __name__ == '__main__':

    try:
        #starts yearly pickle timmer
        start = time.perf_counter()
        
        #accepts command line argument for year
        year = int(sys.argv[1])
        
        pickle_file_name = output_path + 'all_gr_y' + str(year) + '.pickle'
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
