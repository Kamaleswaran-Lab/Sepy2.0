# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 21:44:42 2021
@author: Elite Data Hacks
"""
"""This is a dictionary of aggregation methods to be applied to each lab/vital sign"""

labAGG = {#electrolytes
'anion_gap':'mean', 
'base_excess':'mean', 
'bicarb_(hco3)':'mean', 
'blood_urea_nitrogen_(bun)':'mean',
'calcium':'mean', 
'calcium_adjusted':'mean', 
'calcium_ionized':'mean', 
'chloride':'mean', 
'creatinine':'mean',
'gfr':'mean', 
'glucose':'mean', 
'magnesium':'mean', 
'osmolarity':'mean', 
'phosphorus':'mean', 
'potassium':'mean',
'sodium':'mean',
#CBC
'haptoglobin':'mean', 
'hematocrit':'mean', 
'hemoglobin':'mean', 
'met_hgb':'mean', 
'platelets':'mean',
'white_blood_cell_count':'mean',
'carboxy_hgb':'mean',
#hepatic
'alanine_aminotransferase_(alt)':'mean',
'albumin':'mean', 
'alkaline_phosphatase':'mean',
'ammonia':'mean', 
'aspartate_aminotransferase_(ast)':'mean', 
'bilirubin_direct':'mean',
'bilirubin_total':'mean', 
'fibrinogen':'mean', 
'inr':'mean', 
'lactate_dehydrogenase':'mean', 
'lactic_acid':'mean',
'partial_prothrombin_time_(ptt)':'mean', 
'prealbumin':'mean', 
'protein':'mean', 
'prothrombin_time_(pt)':'mean', 
'thrombin_time':'mean', 
'transferrin':'mean',
#Pancreatic
'amylase':'mean',
'lipase':'mean',
#Cardiac
'b-type_natriuretic_peptide_(bnp)':'mean', 
'troponin':'mean',
#ABG            
'carboxy_hgb':'mean', 
'fio2':'mean', 
'partial_pressure_of_carbon_dioxide_(paco2)':'mean',
'partial_pressure_of_oxygen_(pao2)':'mean', 
'ph':'mean', 
'saturation_of_oxygen_(sao2)':'mean',
#Other            
'd_dimer':'mean', 
'hemoglobin_a1c':'mean', 
'parathyroid_level':'mean', 
'thyroid_stimulating_hormone_(tsh)':'mean',
'lymphocyte': 'mean',
'neutrophils': 'mean',
#Inflammation
'crp_high_sens':'mean', 
'procalcitonin':'mean', 
'erythrocyte_sedimentation_rate_(esr)':'mean',                       
#String Results        
'c_diff':'first', 
'covid':'first', 
'mtp':'first'}