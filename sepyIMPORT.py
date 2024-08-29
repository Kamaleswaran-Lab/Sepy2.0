# -*- coding: utf-8 -*-
"""
Elite Data Hacks
@author: Christopher S. Josef, MD
@email: csjosef@krvmail.com
"""

import os
import time
import pandas as pd
import numpy as np
import pickle

class sepyIMPORT():
    ########################################################################################################
    ####################                  Class Variables                                ###################
    ########################################################################################################

    #for us when importing CSVs
    na_values = ['NULL', 'Date\Time Correction', '(null)','NOTRECORDED' ]

    # vital data type dictionary 
    vital_col_names = ['temperature',
              'daily_weight_kg', 'height_cm',
              'sbp_line', 'dbp_line', 'map_line',
              'sbp_cuff', 'dbp_cuff', 'map_cuff',
              'pulse', 'unassisted_resp_rate', 'spo2', 'end_tidal_co2', 'o2_flow_rate'] 
    
    #Vasopressor units
    vasopressor_units = ['norepinephrine_dose_unit', 
                         'epinephrine_dose_unit', 
                         'dobutamine_dose_unit', 
                         'dopamine_dose_unit', 
                         'phenylephrine_dose_unit', 
                         'vasopressin_dose_unit'] 
    
    #List of all lab names (some years might not have all listed labs)
    numeric_lab_col_names = ['anion_gap', 'base_excess', 'bicarb_(hco3)', 'blood_urea_nitrogen_(bun)',
           'calcium', 'calcium_adjusted', 'calcium_ionized', 'chloride', 'creatinine',
           'gfr', 'glucose', 'magnesium', 'osmolarity', 'phosphorus', 'potassium',
           'sodium',
           
            #CBC  
            'haptoglobin', 'hematocrit', 'hemoglobin', 'met_hgb', 'platelets',
            'white_blood_cell_count','carboxy_hgb',
            
            #Hepatic
            'alanine_aminotransferase_(alt)','albumin', 'alkaline_phosphatase',
            'ammonia', 'aspartate_aminotransferase_(ast)', 'bilirubin_direct', 
            'bilirubin_total', 'fibrinogen', 'inr', 'lactate_dehydrogenase', 'lactic_acid',
            'partial_prothrombin_time_(ptt)', 'prealbumin', 'protein', 
            'prothrombin_time_(pt)', 'thrombin_time', 'transferrin',
            
            #Pancreatic
            'amylase','lipase',
            
            #Cardiac
            'b-type_natriuretic_peptide_(bnp)', 'troponin',
            
            #ABG            
            'carboxy_hgb', 'fio2', 'partial_pressure_of_carbon_dioxide_(paco2)',
            'partial_pressure_of_oxygen_(pao2)', 'ph', 'saturation_of_oxygen_(sao2)',
            
            #Other            
            'd_dimer', 'hemoglobin_a1c', 'parathyroid_level', 'thyroid_stimulating_hormone_(tsh)',
            
            #Inflammation           
            'crp_high_sens', 'procalcitonin', 'erythrocyte_sedimentation_rate_(esr)'  ,  

            #Extra labs
            'neutrophils', 'lymphocyte'
            ]
    
    string_lab_col_names = [
        #PCR Testing
        'c_diff', 'covid', 'mtp'
        ]
    
    all_lab_col_names = numeric_lab_col_names + string_lab_col_names
    
# =============================================================================
#     d_vent_data_types ={'pat_id':int, 'vent_type':object, 'vent_mode':object, 
#                         'vent_rate_set':float, 'vent_tidal_rate_set':float,
#                         'vent_tidal_rate_exhaled':float, 'peep':pd.to_numeric, 'fio2':pd.to_numeric}   
# =============================================================================

    ########################################################################################################
    ####################                  Instance Variables                             ###################
    ########################################################################################################

    def __init__(self, file_dictionary, delim):
        #delimiter used in raw files can changea across data sets; the delim argument is specified by user
        self.delim = delim
        
        # dictionary has file locations for flat files
        self.file_dictionary = file_dictionary
        
        #creates df with all lab groupings
        self.df_grouping_labs = pd.read_csv(file_dictionary['path_grouping_file_labs'])
        
        #creates df with all medication groupings        
        self.df_grouping_all_meds = pd.read_csv(file_dictionary['path_grouping_file_meds'])
        
        #creates df with all bed location labels   
        self.df_bed_labels = pd.read_csv(file_dictionary['path_bed_labels'])     
    #############################################################################################
    ###### Rudimentary data cleaning
    ############################################################################################# 

    def make_numeric(self,
                     df,
                     cols):
        """
        Accepts- 1)a data frame 2)list of cols
        Does- removes text/string artifacts, and converts the columns to numrical (i.e. int, float, etc.).
        Returns- updated version of df with clean, numeric cols
        Notes- All errors are currently coerced
        """

        # Remove all the non-numeric characters from numerical cols
        df[cols] =df[cols].replace(r'\>|\<|\%|\/|\s','',regex=True)

        # Converts specific cols to numeric
        df[cols] = df[cols].apply(pd.to_numeric, errors= 'coerce')
        return (df)
    
    #############################################################################################
    ###### Custom Date Parser to Handle Date Errors (i.e. coerce foolishness)
    #############################################################################################     
    def d_parser(self, s):
        return pd.to_datetime(s, 
                              infer_datetime_format = True, 
                              errors = 'coerce')
    
    
    #############################################################################################
    ###### Create encounter df for all CSNs
    #############################################################################################     
    def import_encounters(self, 
                          drop_cols,
                          index_col,
                          date_cols): 
        
        path_encounters_file= self.file_dictionary['path_encounters_file']
        
        #import file and set date/time cols
        df_encounters = pd.read_csv(path_encounters_file,
                                    header = 0,
                                    index_col = index_col,
                                    parse_dates = date_cols,
                                    infer_datetime_format =True,
                                    na_values = self.na_values,
                                    sep = self.delim,
                                    low_memory = False)

        #drop unecessary cols
        df_encounters = df_encounters.drop(columns=drop_cols)
        

        self.df_encounters = df_encounters
        print('Encounters success')
    #############################################################################################
    ###### Create demographic df for all CSNs
    #############################################################################################      
    def import_demographics(self, 
                            drop_cols,
                            index_col ,
                            date_cols):
        path_demographics_file = self.file_dictionary['path_demographics_file']  
        
        #import files and set date/time cols
        df_demographics = pd.read_csv(path_demographics_file,
                                           header = 0,
                                           index_col = index_col,
                                           parse_dates = date_cols,
                                           infer_datetime_format =True,
                                           na_values = self.na_values,
                                           sep = self.delim,
                                           low_memory = False)
        #drop unecessary cols
        df_demographics = df_demographics.drop(columns = drop_cols)
        
        self.df_demographics = df_demographics
        print('Demographics success')
        
    #############################################################################################
    ###### Create df of filtered medications for all CSNs
    #############################################################################################
    def import_infusion_meds(self,
                             drop_cols,
                             numeric_cols,
                             anti_infective_group_name, 
                             vasopressor_group_name, 
                             index_col,
                             date_cols):
        
        path_infusion_med_file = self.file_dictionary['path_infusion_med_file']

        #imports the infusion med file and sets date/time cols
        read_infusion_csv_time = time.time()
        print('Starting csv read for infusion meds.')
        
        df_infusion_meds = pd.read_csv(path_infusion_med_file,
                                       header = 0, 
                                       parse_dates = date_cols,
                                       date_parser = self.d_parser,
                                       index_col = index_col,
                                       sep = self.delim,
                                       na_values = self.na_values,
                                       low_memory = False,
                                       memory_map = True)
        print(f'It took {time.time()-read_infusion_csv_time} seconds to read the infusion csv.')
        
        df_infusion_meds = self.make_numeric(df_infusion_meds, numeric_cols)
        print('Infusion columns are now numeric')
        
        #drop unecessary columns
        self.df_infusion_meds = df_infusion_meds.drop(columns = drop_cols)
        
        # check if there are duplicate med id's & print warining if there is error
        rows_dropped = self.df_grouping_all_meds.shape[0]-self.df_grouping_all_meds.drop_duplicates('medication_id').shape[0]
        
        if rows_dropped > 0:
            #drop the duplicate rows in grouping all meds
            self.df_grouping_all_meds = self.df_grouping_all_meds.drop_duplicates(subset='medication_id') 
            print(f'You have {rows_dropped} duplicates of medication ID that were dropped!')
            
        else:    
            print('Congrats, You have NO duplicates of medication ID!')
            
        #get med_ids for anti_infective
        df_anti_infective_med_groups = self.df_grouping_all_meds[
            self.df_grouping_all_meds['med_class'] == anti_infective_group_name]\
            [['super_table_col_name','medication_id']]
        

        #makes df with anti-infective meds 
        self.df_anti_infective_meds = df_infusion_meds \
            .reset_index() \
            .merge(df_anti_infective_med_groups, how = "inner", on = "medication_id")  \
            .set_index('csn')
        
        print('Anti-infective success')
        
        #get med_ids for vassopressors
        df_vasopressor_med_groups = self.df_grouping_all_meds[
            self.df_grouping_all_meds['med_class'] == vasopressor_group_name]\
            [['super_table_col_name','medication_id']] 

        # makes df with vasopressor ; adds a numerically increasing index along with csn and supertable name
        # only keeps the dose and dose unit cols   
        df_vasopressor_meds = df_infusion_meds \
            .reset_index() \
            .merge(df_vasopressor_med_groups, how = "inner", on = "medication_id")  \
            .reset_index()\
            .set_index(['csn','med_order_time','super_table_col_name'], append=True) \
            [['med_action_dose','med_action_dose_unit']] 
        
        # unstack the units
        units = df_vasopressor_meds['med_action_dose_unit'].unstack(level=3)

        # unstack the dose
        dose = df_vasopressor_meds['med_action_dose'].unstack(level=3)
        # merge the dose and units together 
        df_vasopressor_meds = dose.merge(units, left_index=True, right_index=True, suffixes=['','_dose_unit' ])
        
        #unstack makes a multi-index for columns; this line removes "lab result" level
        #df_vasopressor_meds.columns = df_vasopressor_meds.columns.droplevel()
        
        #this removes the "name" for all the columns i.e. super_table_col_names
        df_vasopressor_meds.columns.name = None
        
        #removes numerical index for vasopressor meds
        df_vasopressor_meds = df_vasopressor_meds.droplevel(0)
        
        #drops med_order_time as index
        df_vasopressor_meds = df_vasopressor_meds.reset_index(1)
        
        #cols that are have units will break "groupby" related actions later, so need to remove nan
        df_vasopressor_meds[self.vasopressor_units] = df_vasopressor_meds[self.vasopressor_units].replace({np.nan: ''})
        
        self.df_vasopressor_meds = df_vasopressor_meds
        print('Vasopressor success')
        
    #############################################################################################
    ###### Create lab df for all CSNs
    #############################################################################################  
    def import_labs(self, 
                    drop_cols,
                    group_cols,
                    date_cols,
                    index_col,
                    numeric_cols):
        ### The labs import needs a special function to tiddy the columns
        def tidy_index(df):
            #turns super_table_name into a col
            df = df.unstack(level=1)
            #unstack makes a multi-index for columns; this line removes "lab result" level
            df.columns = df.columns.droplevel()

            #this removes the "name" for all the columns i.e. super_table_col_names
            df.columns.name = None
            
            #removes numerical index for labs
            df = df.droplevel(0)
        
            return df
        
        #start timer for the lab import function
        start_import_time = time.time()
        print('Begin Lab Import')
        #Lab groups file has three important cols: 
            #1) component_id - the is the id number for the lab type 
            #2 )super_table_col_name- is the group name for similar labs
            #3) physionet - indicates if used for physionet competition
            
        lab_groups = self.df_grouping_labs[group_cols]
        
        #### Import Lab Data file #####
        #set path variables based on file_dictionary
        path_lab_file = self.file_dictionary['path_lab_file']

        #import the lab flat file and set date/time
        df_labs = pd.read_csv(path_lab_file,
                              header = 0,
                              parse_dates = date_cols,
                              date_parser = self.d_parser,
                              na_values = self.na_values,
                              sep = self.delim,
                              low_memory = False,
                              memory_map = True)
        
        # drop unecessary cols
        df_labs = df_labs.drop(columns = drop_cols)
                                     
        ### Select Relavent Lab Groups ###
        lab_groups = lab_groups[['super_table_col_name','component_id']][lab_groups['import'] == "Yes"]
    
        ### Join Groups and Lab File ###
        df_labs_filtered = df_labs.merge(lab_groups,
                                    how = "inner",
                                    on = "component_id")
        
        #if there is no collection time, use result time - 1hr
        df_labs_filtered['collection_time'] = df_labs_filtered['collection_time'].fillna(df_labs_filtered['lab_result_time']- pd.Timedelta(hours = 1))
        
        # set index (necessary for unstacking)
        df_labs_filtered.set_index(['super_table_col_name',
                                  'csn',
                                  'component_id',
                                  'result_status',
                                  'lab_result_time',
                                  'collection_time',
                                  'pat_id',
                                  'proc_cat_id',
                                  'proc_cat_name',
                                  'proc_code',
                                  'proc_desc',
                                  'component',
                                  'loinc_code'], \
                                  append=True, inplace=True)
        
        ####### Select Labs that have string value ##########
        #isolate string lab value rows
        df_labs_filtered_string = df_labs_filtered.loc[df_labs_filtered.index.get_level_values('super_table_col_name').isin(self.string_lab_col_names)]
        #####################################################
        
        ####### Select and Treat Labs that have Numeric value ##########
        #isolate numeric lab value rows
        df_labs_filtered_numeric = df_labs_filtered.loc[df_labs_filtered.index.get_level_values('super_table_col_name').isin(self.numeric_lab_col_names)]
        
        #remove silly punctuation from numeric
        df_labs_filtered_numeric = df_labs_filtered_numeric.replace(r'\>|\<|\%|\/|\s','',regex=True)

        #convert labs to numeric
        df_labs_filtered_numeric['lab_result'] = pd.to_numeric(df_labs_filtered_numeric['lab_result'], errors='coerce')    
        #####################################################
        
        #Tiddy up index using previously defined fcn        
        df_labs_numeric = tidy_index(df_labs_filtered_numeric)
        df_labs_string = tidy_index(df_labs_filtered_string )
        
        #print time it takes to import and unstack labs
        print(f'It took {time.time() - start_import_time}(s) to import and process labs.')
        
        df_labs_numeric = df_labs_numeric
        df_labs_string = df_labs_string
        # Concat the string and numeric dfs
        df_labs_all = pd.concat([df_labs_numeric, df_labs_string], axis=0)
        
        #if there are missing cols (i.e. no COVID in 2014) then it ensures the col name is added
        self.df_labs = df_labs_all.reindex(df_labs_all.columns.union(self.all_lab_col_names),axis=1)  
        print('Labs success')

    #############################################################################################
    ###### Create vitals df for all CSNs
    #############################################################################################  
    def import_vitals(self, 
                      drop_cols,
                      numeric_cols,
                      index_col,
                      date_cols,
                      merge_cols):
        
        path_vitals_file = self.file_dictionary['path_vitals_file']

        #import vitals flat file and set date/time cols
        df_vitals = pd.read_csv(path_vitals_file,
                                header = 0,
                                dtype = object,
                                parse_dates = date_cols,
                                infer_datetime_format =True,
                                index_col = index_col,
                                na_values = self.na_values,
                                sep = self.delim,
                                low_memory = False)
        #drop unecssary cols
        df_vitals = df_vitals.drop(columns=drop_cols)
        
        # If there are columns to merge then do this:
        if merge_cols is not None:
            for merge_set in merge_cols:
                df_vitals[merge_set[2]] = df_vitals[merge_set[0]].fillna(df_vitals[merge_set[1]])
                df_vitals = df_vitals.drop(columns = [merge_set[0], merge_set[1]])
        
        #drop punctuation and make numeric
        start_to_numeric_conversion_time = time.time() # start timer
        df_vitals = self.make_numeric(df_vitals, numeric_cols)
        print(f'It took {time.time()-start_to_numeric_conversion_time} to convert vitals results to numeric.')
                
        self.df_vitals = df_vitals
        print('Vitals success')
    #############################################################################################
    ###### Create vent df for all CSNs
    #############################################################################################      
    def import_vent(self,
                    drop_cols,
                    numeric_cols,
                    index_col,
                    date_cols):
        
        path_vent_file = self.file_dictionary['path_vent_file']
        
        #import vent flat files and set date/time
        df_vent = pd.read_csv(path_vent_file,
                              header = 0,
                              parse_dates = date_cols,
                              infer_datetime_format = True,
                              index_col=index_col,
                              na_values = self.na_values,
                              sep = self.delim,
                              low_memory = False)
        
        #drop unecessary cols
        df_vent = df_vent.drop(columns = drop_cols)
        
        df_vent = self.make_numeric(df_vent, numeric_cols)
        
        self.df_vent = df_vent
        print('Vent success')
    #############################################################################################
    ###### Create GCS df for all CSNs
    #############################################################################################     
    def import_gcs(self,
                   drop_cols,
                   index_col,
                   numeric_col,
                   date_cols):
        
        path_gcs_file = self.file_dictionary['path_gcs_file']   
        
        #import gcs flat file and set date/time cols
        df_gcs = pd.read_csv(path_gcs_file,
                             header = 0, 
                             parse_dates = date_cols,
                             infer_datetime_format = True,
                             index_col = index_col,
                             na_values = self.na_values,
                             sep = self.delim,
                             low_memory = False)
        
        #drop unecessary cols
        df_gcs = df_gcs.drop(columns=drop_cols)
        
        #ensure all score coloumns are numeric
        self.make_numeric(df_gcs, numeric_col)
        
        #merges all gcs values into a single timestamp/row
        df_gcs = df_gcs.groupby(['csn','recorded_time']) \
           .aggregate({'gcs_eye_score':['mean'],
                       'gcs_verbal_score':['mean'],
                       'gcs_motor_score':['mean'],
                       'gcs_total_score':['mean']})      
           
        #drops the column index "mean" which came from agg fcn
        #also moves 'recorded time' out of index into a column
        df_gcs = df_gcs.droplevel(1,axis=1).reset_index(level='recorded_time')
           
        self.df_gcs = df_gcs
        print('GCS success')
    #############################################################################################
    ###### Create cultures df for all CSNs
    #############################################################################################        
    def import_cultures(self, 
                        drop_cols,
                        index_col,
                        date_cols):
        
        path_cultures_file = self.file_dictionary['path_cultures_file']      
        
        #imports cultures and sets date/time cols
        df_cultures = pd.read_csv(path_cultures_file,
                                  header = 0,
                                  parse_dates = date_cols,
                                  infer_datetime_format = True,
                                  index_col = index_col,
                                  na_values = self.na_values,
                                  sep = self.delim,
                                  low_memory = False) 
        
        #drops unecessary cols        
        df_cultures = df_cultures.drop(columns = drop_cols)
          
        self.df_cultures = df_cultures
        print('Cultures success')
        
    #############################################################################################
    ###### Create bed_location df for all CSNs
    #############################################################################################     
    def import_bed_locations(self, 
                             drop_cols,
                             index_col,                             
                             date_cols):
        
        bed_labels = self.df_bed_labels
        path_bed_locations_file = self.file_dictionary['path_bed_locations_file']
  
        #import bed flat files and set date/time cols
        df_beds = pd.read_csv(path_bed_locations_file,
                              header = 0,
                              parse_dates = date_cols,
                              infer_datetime_format =True,
                              index_col = index_col,
                              na_values = self.na_values,
                              sep = self.delim,
                              low_memory = False)
        
        #drop anytimes where bed_location_start= bed_location_end
        df_beds = df_beds[df_beds['bed_location_start'] != df_beds['bed_location_end']]
        
        # Identifier column for ICU bed
        icu_units = bed_labels[bed_labels['icu']==1].bed_unit.tolist()
        df_beds['icu'] = np.where(df_beds['bed_unit'].isin (icu_units),1,0)
        
        # Identifier column for IMC
        imc_units = bed_labels[bed_labels['imc']==1].bed_unit.tolist()
        df_beds['imc'] = np.where(df_beds['bed_unit'].isin (imc_units),1,0)
        
        # Identifier column for ED bed
        ed_units = bed_labels[bed_labels['ed']==1].bed_unit.tolist()
        df_beds['ed'] = np.where(df_beds['bed_unit'].isin (ed_units),1,0)
        
        # Identifier column for peri procedure bed
        procedure_units = bed_labels[bed_labels['procedure']==1].bed_unit.tolist()
        df_beds['procedure'] = np.where(df_beds['bed_unit'].isin (procedure_units),1,0)
        
        #Get rid of duplicate rows
        df_beds=df_beds.groupby(['csn','pat_id','bed_location_start']).first().reset_index(level=(1,2))
                
        #drop unecessary cols
        df_beds = df_beds.drop(columns = drop_cols)
        
        self.df_beds = df_beds
        print('Beds success')

    #############################################################################################
    ###### Create procedures df for all CSNs
    #############################################################################################     
    def import_procedures (self, 
                           drop_cols,
                           index_col,
                           date_cols):
        path_procedures_file = self.file_dictionary['path_procedures_file'] 
        
        #import procedure flat file and set date/time cols
        df_procedures = pd.read_csv(path_procedures_file,
                                    header = 0,
                                    parse_dates = date_cols,
                                    infer_datetime_format =True,
                                    index_col = index_col,
                                    na_values = self.na_values,
                                    sep = self.delim,
                                    low_memory = False)

        #drop unecessary cols
        df_procedures = df_procedures.drop(columns=drop_cols) 

        self.df_procedures = df_procedures
        print('Procedures success')
    #############################################################################################
    ###### Import ICD 9/10 Code df for all CSNs
    #############################################################################################        
    def import_diagnosis(self, 
                         drop_cols,
                         index_col,
                         date_cols):
        path_diagnosis_file = self.file_dictionary['path_diagnosis_file'] 
        
        #import diganosis flat file
        df_diagnosis = pd.read_csv(path_diagnosis_file,
                                   header = 0, 
                                   parse_dates = date_cols,
                                   infer_datetime_format =True,
                                   index_col = index_col,
                                   na_values = self.na_values,
                                   sep = self.delim,
                                   low_memory=False)
        
        #drop unecessary cols
        df_diagnosis = df_diagnosis.drop(columns=drop_cols) 
                
        self.df_diagnosis = df_diagnosis
        print('Diagnosis success')
        
#### ICD9 Portion
# =============================================================================
#         self.df_ahrq_ICD9 = self.make_comorbid_df(self.file_dictionary['path_comorbid_ahrq_ICD9'],
#                                         'ICD9',
#                                         'ahrq',
#                                         'dx_code_icd9',
#                                         'v_ahrq_labels')
#         # makes df for ahrq (like elix)
#         self.df_elix_ICD9 = self.make_comorbid_df(self.file_dictionary['path_comorbid_elix_ICD9'],
#                                         'ICD9',
#                                         'elix',
#                                         'dx_code_icd9',
#                                         'v_elix_labels')
#         # makes df for Charlson        
#         self.df_quan_deyo_ICD9 = self.make_comorbid_df(self.file_dictionary['path_comorbid_quan_deyo_ICD9'],
#                                         'ICD9',
#                                         'quan_deyo',
#                                         'dx_code_icd9',
#                                         'v_quan_deyo_labels')
#         # makes df for Quan's Elix 
#         self.df_quan_elix_ICD9 = self.make_comorbid_df(self.file_dictionary['path_comorbid_quan_elix_ICD9'],
#                                 'ICD9',
#                                 'quan_elix',
#                                 'dx_code_icd9',
#                                 'v_quan_elix_labels')
#         # makes df for ccs
#         self.df_ccs_ICD9 = self.make_comorbid_df(
#                                 self.file_dictionary['path_comorbid_ccs_ICD9'],
#                                 'ICD9',
#                                 'ccs_label',
#                                 'dx_code_icd9',
#                                 'v_ccs_labels')
# =============================================================================

#### ICD10 Portion
# =============================================================================
#         self.df_ahrq_ICD10 = self.make_comorbid_df(self.file_dictionary['path_comorbid_ahrq_ICD10'],
#                                 'ICD10',
#                                 'ahrq',
#                                 'dx_code_icd10',
#                                 'v_ahrq_labels')
#         # makes df for ahrq (like elix)
#         self.df_elix_ICD10 = self.make_comorbid_df(self.file_dictionary['path_comorbid_elix_ICD10'],
#                                 'ICD10',
#                                 'elix',
#                                 'dx_code_icd10',
#                                 'v_elix_labels')
# =============================================================================
        # makes df for Charlson        
        self.df_quan_deyo_ICD10 = self.make_comorbid_df(self.file_dictionary['path_comorbid_quan_deyo_ICD10'],
                                'ICD10',
                                'quan_deyo',
                                'dx_code_icd10',
                                'v_quan_deyo_labels')
        # makes df for Quan's Elix 
        self.df_quan_elix_ICD10 = self.make_comorbid_df(self.file_dictionary['path_comorbid_quan_elix_ICD10'],
                                'ICD10',
                                'quan_elix',
                                'dx_code_icd10',
                                'v_quan_elix_labels')
# =============================================================================
#         # makes df for ccs
#         self.df_ccs_ICD10 = self.make_comorbid_df(
#                                 self.file_dictionary['path_comorbid_ccs_ICD10'],
#                                 'ICD10',
#                                 'ccs_label',
#                                 'dx_code_icd10',
#                                 'v_ccs_labels')
# =============================================================================
        print('Comorbid success')
        
    def make_comorbid_df(self, 
                         comorbid_map_path,
                         map_ICD_col,
                         map_comorbidity_col,
                         df_diagnosis_ICD_col,
                         comorbid_labels):
        #import mapping file
        map_df = pd.read_csv(comorbid_map_path, 
                             header = 0)
        #column names in map file
        map_df_col_names = map_df.columns.to_list()
        
        # creates a variable with all the comorbidity types
        setattr(self, 
                comorbid_labels, 
                map_df[map_comorbidity_col].unique().tolist())
        
        #merge mapping file with diagnosis file
        all_diagnoses = self.df_diagnosis
        all_diagnoses[df_diagnosis_ICD_col] = all_diagnoses[df_diagnosis_ICD_col].astype(str).str.replace('.','', regex=False)#.drop_duplicates()
        mapped = map_df.merge(all_diagnoses.reset_index(), 
                             how='left', 
                             left_on = map_ICD_col,
                             right_on = df_diagnosis_ICD_col)\
                             .set_index('csn')
        return mapped[['pat_id','dx_time_date'] + map_df_col_names]

        