import pandas as pd
import os
from pathlib import Path
path_to_pickles = Path('/labs/collab/K-lab-MODS/MODS-PHI/Encounter_Pickles/emHolder_OutlierCorrected/')

year = 2014 

# Get all the files in the directory
files = list(path_to_pickles.glob(f'*{year}_csvs/*.csv'))

#Write a function to read the csv file, add a column with the filename, add a column with the row counter 
#and return the dataframe
def read_csv(file):
    df = pd.read_csv(file)
    df['encounter'] = file.split('/')[-1].split('.')[0]
    df['row_counter'] = range(len(df))
    return df


# Combine all the files into one dataframe
df = pd.concat([read_csv(file) for file in files])

df.to_csv(f'/labs/collab/K-lab-MODS/MODS-PHI/Encounter_Pickles/emHolder_OutlierCorrected/combined_files_{year}.csv', index=False)