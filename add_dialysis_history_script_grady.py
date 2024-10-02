# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 15:36:24 2023

@author: mehak

TODO: This was done in a hurry, needs to be cleaned up.

This script is meant to be run AFTER the script that creates the super tables. 
It adds a column to the super tables that indicates whether the patient has a history of dialysis. Also corrects the 
column names of the super tables to remove the time columns. 
TODO: Add this code directly to the script that creates the super tables.

BUT, it is useful to have a layer that runs on top of the super tables to add additional information. ??
Computationaly cheaper than creating all the supertables again. 
"""

import concurrent.futures
import time
import pandas as pd
import numpy as np
import os
import glob
import pickle
import sys

def add_dialysis_history(encounter, new_sup_path, year):
    try:
        with open(os.path.join(new_sup_path, year, encounter), 'rb') as f:
            supertable = pickle.load(f)
        time_columns = [x for x in supertable['super_table'].columns.tolist() if (x.endswith('collection_time') or x.endswith('recorded_time'))]
        if len(time_columns) > 0:
            supertable['super_table'] = supertable['super_table'].drop(columns = time_columns)
    except Exception as e:
        print("Error in encounter {}".format(encounter))
        print(e)
        return
    
    try:
        dialysis_history = supertable['diagnosis_PerCSN'].loc[(supertable['diagnosis_PerCSN'].dx_code_icd9 == '585.6') | (supertable['diagnosis_PerCSN'].dx_code_icd10 == 'N18.6')]
        if len(dialysis_history) == 0:
            supertable['super_table']['history_of_dialysis'] = [0]*len(supertable['super_table'])
        else:
            supertable['super_table']['history_of_dialysis'] = [1]*len(supertable['super_table'])

        new_save_path = os.path.join(new_sup_path, year + '_modified')
        with open(os.path.join(new_save_path, encounter), 'wb') as f:
            pickle.dump(supertable, f)

        csv_path = os.path.join(new_sup_path, year + '_csvs')
        supertable['super_table'].to_csv(os.path.join(csv_path, encounter.split('.')[0] + '.csv'))
        
        print("Modification completed for encounter {}".format(encounter))
    except Exception as e:
        print("Error in modifying encounter {}".format(encounter))
        print(e)
        

    
if __name__ == "__main__":
    
    year = sys.argv[1]
    new_sup_path = '/labs/collab/K-lab-MODS/MODS-PHI/Encounter_Pickles/grHolder/'
    encounters = os.listdir(os.path.join(new_sup_path, year))
    
    if not os.path.exists(os.path.join(new_sup_path, year + '_modified')):
        os.mkdir(os.path.join(new_sup_path, year + '_modified'))
    
    if not os.path.exists(os.path.join(new_sup_path, year + '_csvs')):   
        os.mkdir(os.path.join(new_sup_path, year + '_csvs'))
    
    print(pd.__version__)
    
    def pool_function(i):
        encounter = encounters[i]
        add_dialysis_history(encounter, new_sup_path, year)
        
    
    with concurrent.futures.ProcessPoolExecutor(16) as executor:
        start_time = time.perf_counter()
        list(executor.map(pool_function, range(len(encounters))))
        finish_time = time.perf_counter()
    
    print("Time : {} s".format(finish_time - start_time))

