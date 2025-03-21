import pandas as pd
import hashlib
from pathlib import Path
from argparse import ArgumentParser

hash_key = '123'

YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]

def hash_value(value, hash_key):
    return hashlib.sha256((str(value) + hash_key).encode()).hexdigest()

identifier_columns = ['pat_id', 'csn']
drop_columns = ['first_name', 'last_name', 'mi', 'last4_ssn'] 

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--index', type = int, required = True)
    args = parser.parse_args()
    year = YEARS[args.index]

    path_to_data = Path('/labs/collab/K-lab-MODS/MODS-PHI/Grady_Data/')
    path_to_deid_data = Path('/labs/kamaleswaranlab/MODS/Data/deid_grady/')
    path_to_lists = Path('/labs/collab/K-lab-MODS/MODS-PHI/Grady_Data/1. Administrative Attributes')
    path_to_deid_data = path_to_deid_data / str(year)
    path_to_deid_data.mkdir(parents = True, exist_ok = True)

    files = list(path_to_data.glob(f'*/*/*_{year}_*.txt'))
    
    for file in files:
        print(file)
        if 'clinical_notes' in file.name:
            print(file.name + ' skipped')
            continue
        df = pd.read_csv(file, sep = '|', error_bad_lines = False)
        
        if 'encounter' in file.name:
            matching_list = pd.DataFrame( columns= ['pat_id', 'pat_id_deid'])
            matching_list['pat_id'] = df['pat_id'].unique()
            matching_list['pat_id_deid'] = matching_list['pat_id'].apply(lambda x: hash_value(x, hash_key))
            matching_list.to_csv(path_to_lists / f'matching_list_patid_{year}.csv', index = False)
            print("Pat ID matching list saved")   
            
            matching_list = pd.DataFrame( columns= ['csn', 'csn_deid'])
            matching_list['csn'] = df['csn'].unique()
            matching_list['csn_deid'] = matching_list['csn'].apply(lambda x: hash_value(x, hash_key))
            matching_list.to_csv(path_to_lists / f'matching_list_csn_{year}.csv', index = False)
            print("CSN matching list saved")

            matching_list = pd.DataFrame( columns= ['har', 'har_deid'])
            matching_list['har'] = df['har'].unique()
            matching_list['har_deid'] = matching_list['har'].apply(lambda x: hash_value(x, hash_key))
            matching_list.to_csv(path_to_lists / f'matching_list_csn_{year}.csv', index = False)
            print("HAR matching list saved")

            matching_list = pd.DataFrame( columns= ['mrn', 'mrn_deid'])
            matching_list['mrn'] = df['mrn'].unique()
            matching_list['mrn_deid'] = matching_list['mrn'].apply(lambda x: hash_value(x, hash_key))
            matching_list.to_csv(path_to_lists / f'matching_list_csn_{year}.csv', index = False)
            print("MRN matching list saved")

            try:
                df.drop(columns=['zip_code'], inplace = True)
            except KeyError:
                df.drop(columns = ['ZIP_CODE'], inplace = True)
            print("Zip code column dropped")

        # Deidentify identifier columns
        for column in identifier_columns:
            try:
                if column in df.columns:
                    df[column] = df[column].apply(lambda x: hash_value(x, hash_key))
                else:
                    column = column.upper()
                    df[column] = df[column].apply(lambda x: hash_value(x, hash_key))
            except KeyError:
                print(column + ' does not exist in ' + file.name)
        
        print(file.name + " deidentified")
        if 'DEMOGRAPHICS' in file.name:
            #Drop Columns
            try:
                df.drop(columns=drop_columns, inplace = True)
            except KeyError:
                drop_columns = [column.upper() for column in drop_columns]
                df.drop(columns = drop_columns, inplace = True)
            print("Demographic columns dropped")

        df.to_csv(path_to_deid_data / file.name, index = False, sep = '|')
        del df 

    
