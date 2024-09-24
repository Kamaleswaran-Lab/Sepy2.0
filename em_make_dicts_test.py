# -*- coding: utf-8 -*-
"""
Created on Wed Sept 24 13:22:04 2024

@author: mehak
"""
import sys
import pandas as pd
import numpy as np
import sepyDICT as sd
import pickle
import time
from pathlib import Path

bed_unit_csv_fname = Path('/labs/collab/K-lab-MODS/EliteDataHacks/Sepy/bed_units_to_icu_AH.csv')
variable_bounds_csv_fname = Path('/labs/collab/K-lab-MODS/EliteDataHacks/Sepy/Variable_Chart.xlsx')
dialysis_info_csv_fname = Path('/labs/collab/K-lab-MODS/EliteDataHacks/Sepy/PEACH_HD_CRRT(1).csv')

def process_csn(csn, pickle_write_path, bed_to_unit_mapping, bounds, dialysis_year):
    
    file_name = pickle_write_path / (str(csn) + '.pickle')
    
    #instantiate class for single encounter
    instance = sd.sepyDICT(yearly_instance, csn, bed_to_unit_mapping, bounds, dialysis_year )
    #create encounter dictionary
    inst_dict = instance.encounter_dict
        
    #create a pickle file for encounter
    picklefile = open(file_name, 'wb')
    #pickle the encounter dictionary and write it to file
    pickle.dump(inst_dict, picklefile)
    #close the file
    picklefile.close()
    
    #return dictionary for summary report functions
    return(instance)

if __name__ == "__main__":
    csn_list_file_name = Path("/labs/collab/K-lab-MODS/EliteDataHacks/pt_lists/pt_list_2014.psv") #imports file location for CSN list
    print(f'MkDct- The CSN List location: {csn_list_file_name}')
    
    pickle_path = Path("/labs/collab/K-lab-MODS/Yearly_Pickles/HolderSuperTables/") #sets directory for yearly pickles
    print(f'MkDct- The pickle directory: {pickle_path}')
    
    output_path =  Path("/labs/collab/K-lab-MODS/MODS-PHI/Encounter_Pickles/emHolder_OutlierCorrected/") #sets directory for output
    print(f'MkDct- The output directory: {output_path}')


    bash_year = int(2014) # year num provided in bash
    print(f'MkDct- The import year is: {bash_year}')

    # Cohort selector
    ed = 0 
    in_pt = 1 
    icu = 0 
    adult = 1
    vent_row = 0 
    vent_start = 0
    
    # reads the list of csns
    csn_df = pd.read_csv(csn_list_file_name, sep='|', header=0)

    #  only keep csns that meet specified year
    csn_df = csn_df[(csn_df.in_pt == in_pt) & (csn_df.adult == adult)]
    print(f'MkDct- A total of {csn_df.shape[0]} encounters were selected')
    csn_df = csn_df[['csn','year']]
    
    # drop duplicates
    csn_df = csn_df.drop_duplicates()
    csn_df = csn_df[csn_df.year == bash_year]
    total_num_enc = len(csn_df)
    print(f'MkDct- The year {bash_year} has {total_num_enc} encounters.')

    #select correct pickle by year
    pickle_name = pickle_path / ('em_y' + str(bash_year) + '.pickle')
    print(f'MkDct- The following pickle is being read: {pickle_name}')
        
    # reads the IMPORT class instance (i.e.  1 year of patient data)
    pickle_load_time = time.perf_counter() #time to load pickle
    #with open(pickle_name, 'rb') as handle:
    #    yearly_instance = pickle.load(handle)

    yearly_instance = pd.read_pickle(pickle_name)
    print(f'MkDct-Pickle from year {bash_year} was loaded in {time.perf_counter()-pickle_load_time}s.')

    print("-----------LOADED YEARLY PICKLE FILE!!!!---------------")

    # if success, make a dir for this year's encounters
    pickle_write_path = output_path / str(bash_year)
    Path.mkdir(pickle_write_path, exist_ok = True)
    print(f'MkDct-Directory for year {bash_year} was set to {pickle_write_path}')
        
    # make empty list to handle csn's with errors
    error_list = []
    start_csn_creation = time.perf_counter() #times calc's by year
        
    ############ LOAD FILES FOR EXTRA PROCESSING ####
    
    bed_to_unit_mapping = pd.read_csv(bed_unit_csv_fname)
    bed_to_unit_mapping.drop(columns = ['Unnamed: 0'], inplace = True)
    bed_to_unit_mapping.columns = ['bed_unit', 'icu_type', 'unit_type', 'hospital']
    
    bounds = pd.read_excel(
            variable_bounds_csv_fname,
            engine='openpyxl',
    )
    
    dialysis = pd.read_csv(dialysis_info_csv_fname)
    dialysis_year = dialysis.loc[dialysis['Encounter Encounter Number'].isin(csn_df['csn'].values)]

    for idx, csn in enumerate(csn_df['csn']):

        print(f'MkDct- The current pt csn is: {csn}, which is {idx + 1} of {len(csn_df)} for year {bash_year}')
        import pdb; pdb.set_trace()
        instance = process_csn(csn, pickle_write_path, bed_to_unit_mapping, bounds, dialysis_year)

      

