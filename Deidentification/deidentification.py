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
    print(year)
    path_to_data = Path('/labs/collab/K-lab-MODS/MODS-PHI/Emory_Data/')
    path_to_deid_data = Path('/labs/kamaleswaranlab/MODS/Data/deid/')

    path_to_data = path_to_data / str(year)
    path_to_deid_data = path_to_deid_data / str(year)
    path_to_deid_data.mkdir(parents = True, exist_ok = True)

    files = path_to_data.glob('*.csv')
    for file in files:
        df = pd.read_csv(file)
        if 'ENCOUNTER' in file.name:
            try:
                matching_list = pd.DataFrame( columns= ['pat_id', 'pat_id_deid'])
                matching_list['pat_id'] = df['pat_id'].unique()
                matching_list['pat_id_deid'] = matching_list['pat_id'].apply(lambda x: hash_value(x, hash_key))
                matching_list.to_csv(path_to_data / 'matching_list_patid.csv', index = False)
                print("Pat ID matching list saved")   
                matching_list = pd.DataFrame( columns= ['csn', 'csn_deid'])
                matching_list['csn'] = df['csn'].unique()
                matching_list['csn_deid'] = matching_list['csn'].apply(lambda x: hash_value(x, hash_key))
                matching_list.to_csv(path_to_data / 'matching_list_csn.csv', index = False)
                print("CSN matching list saved")
            except KeyError:
                matching_list = pd.DataFrame( columns= ['pat_id', 'pat_id_deid'])
                matching_list['pat_id'] = df['PAT_ID'].unique()
                matching_list['pat_id_deid'] = matching_list['pat_id'].apply(lambda x: hash_value(x, hash_key))
                matching_list.to_csv(path_to_data / 'matching_list_patid.csv', index = False)
                print("Pat ID matching list saved")   
                matching_list = pd.DataFrame( columns= ['csn', 'csn_deid'])
                matching_list['csn'] = df['CSN'].unique()
                matching_list['csn_deid'] = matching_list['csn'].apply(lambda x: hash_value(x, hash_key))
                matching_list.to_csv(path_to_data / 'matching_list_csn.csv', index = False)
                print("CSN matching list saved")

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
        print(path_to_deid_data / file.name)
        del df 

    
