import pandas as pd
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool

#Columns to drop
cols = ['on_vent_old', 'vent_fio2_old', 
 'bed_unit']

path_to_supertable = Path('/labs/collab/K-lab-MODS/MODS-PHI/Encounter_Pickles/emHolder_OutlierCorrected')
path_to_deid_supertable = Path('/labs/collab/K-lab-MODS/MODS-PHI/Encounter_Pickles/deid_supertables/')

def get_files(year):
    #Get the list of files
    files = list(path_to_supertable.glob( f'{year}_modified' + '*.pkl'))
    return files   

def create_mapping(files, year):
    #Assign serial numbers from 0 to len(files) randomly to each file
    random_serials = np.random.permutation(len(files))
    mapping = pd.DataFrame({'file': files, 'serial': random_serials})
    #Save the mapping
    mapping.to_csv(path_to_deid_supertable / f'mapping_{year}' + '.csv', index = False)
    return mapping

def check_age(supertable):
    #Check if the age of the person is greater than 80
    if supertable['age'].max() > 80:
        supertable['age'] = [80]*len(supertable)
    return supertable

def drop_dates(supertable):
    #Drop the index 
    supertable = supertable.reset_index().drop(columns = ['index'])
    return supertable

def drop_cols(supertable, cols):
    supertable.drop(columns = cols, inplace = True)
    carboxy = supertable['carboxy_hgb'].values[:,0].shape
    supertable.drop(columns = ['carboxy_hgb'], inplace = True)
    supertable['carboxy_hgb'] = carboxy
    return supertable

def deid_encounter(supertable_path, mapping, path_to_deid_supertable):
    #Read the supertable
    supertable = pd.read_pickle(supertable_path)['super_table']
    #Get the mapping value
    mapping_val = mapping.loc[mapping['file'] == supertable_path, 'serial'].values[0]
    #Check if the age is greater than 80
    supertable = check_age(supertable)
    #Drop the dates
    supertable = drop_dates(supertable)
    #Drop the columns
    supertable = drop_cols(supertable, cols)
    #Save the deidentified supertable
    supertable.to_csv(path_to_deid_supertable / str(year) /  str(mapping_val) + '.csv')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--year', type = int, required= True, help = 'Year for which the deidentification is to be done')
    args = parser.parse_args()
    year = args.year

    path_to_deid_supertable = path_to_deid_supertable / str(year)
    path_to_deid_supertable.mkdir(parents = True, exist_ok = True)

    #Get the list of files
    files = get_files(year)
    #Create the mapping
    mapping = create_mapping(files, year)
    #Deidentify the supertables
    with Pool() as p:
        p.starmap(deid_encounter, [(file, mapping, path_to_deid_supertable) for file in files])
    
    print(f'Deidification for year {year} done')
    
    
