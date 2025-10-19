"""
This script converts the pickled supertables into csv files with a certain set of columns 
(for Dr. Holder's sepsis projects)
"""

import pandas as pd 
import numpy as np
from pathlib import Path 
import os 
import argparse

SUPERTABLE_FEATURES_MAP = {
    'age': 'age',
    'gender': 'gender',
    'race': 'race',
    'ethnicity': 'ethnicity',
    'cci9': 'cci_icd9',
    'cci10': 'cci_icd10',
    'height_cm': 'height_cm',

    'daily_weight_kg': 'daily_weight_kg',

    'pulse_pressure': 'pulse_pressure',
    'best_map': 'best_map',
    'pulse': 'pulse',
    'unassisted_resp_rate': 'unassisted_resp_rate',
    'spo2': 'spo2',
    'temperature': 'temperature',
    'end_tidal_co2': 'end_tidal_co2',
    'gcs_eye_score': 'gcs_eye_score',
    'gcs_verbal_score': 'gcs_verbal_score',
    'gcs_motor_score': 'gcs_motor_score',
    'gcs_total_score': 'gcs_total_score',

    'anion_gap': 'anion_gap',
    'base_excess': 'base_excess',
    'bicarb_(hco3)': 'bicarb_(hco3)',
    'blood_urea_nitrogen_(bun)': 'blood_urea_nitrogen_(bun)',
    'calcium': 'calcium',
    'calcium_ionized': 'calcium_ionized',
    'chloride': 'chloride',
    'creatinine': 'creatinine',
    'gfr': 'gfr',
    'glucose': 'glucose',
    'magnesium': 'magnesium',
    'osmolarity': 'osmolarity',
    'phosphorus': 'phosphorus',
    'potassium': 'potassium',
    'sodium': 'sodium',
    'aspartate_aminotransferase_(ast)': 'aspartate_aminotransferase_(ast)',
    'alanine_aminotransferase_(alt)': 'alanine_aminotransferase_(alt)',
    'bilirubin_direct': 'bilirubin_direct',
    'bilirubin_total': 'bilirubin_total',
    'albumin': 'albumin',
    'alkaline_phosphatase': 'alkaline_phosphatase',
    'prealbumin': 'prealbumin',
    'protein': 'protein',
    'lactate_dehydrogenase': 'lactate_dehydrogenase',

    'haptoglobin': 'haptoglobin',
    'hematocrit': 'hematocrit',
    'hemoglobin': 'hemoglobin', #TODO: ask between this and hemoglobin_a1c
    'platelets': 'platelets',
    'white_blood_cell_count': 'white_blood_cell_count',
    'n_to_l': 'n_to_l',

    'fibrinogen': 'fibrinogen',
    'inr': 'inr',
    'partial_prothrombin_time_(ptt)': 'partial_prothrombin_time_(ptt)',
    'prothrombin_time_(pt)': 'prothrombin_time_(pt)',
    'd_dimer': 'd_dimer',
    'thrombin_time': 'thrombin_time',

    'oxygen_flow_rate': 'o2_flow_rate',
    'fio2': 'fio2',
    'vent_fio2': 'vent_fio2',
    'partial_pressure_of_carbon_dioxide_(paco2)': 'partial_pressure_of_carbon_dioxide_(paco2)',
    'partial_pressure_of_oxygen_(pao2)': 'partial_pressure_of_oxygen_(pao2)',
    'ph': 'ph',
    'saturation_of_oxygen_(sao2)': 'saturation_of_oxygen_(sao2)',
    'met_hgb': 'met_hgb',
    'carboxy_hgb': 'carboxy_hgb',
    's2f_vent_fio2': 'pf_sp',
    'p2f_vent_fio2': 'pf_pa',

    'transferrin': 'transferrin',
    'lactic_acid': 'lactic_acid',
    'ammonia': 'ammonia',
    'amylase': 'amylase',
    'lipase': 'lipase',
    'b-type_natriuretic_peptide_(bnp)': 'b-type_natriuretic_peptide_(bnp)',
    'troponin': 'troponin',
    'hemoglobin_a1c': 'hemoglobin_a1c',
    'parathyroid_level': 'parathyroid_level',
    'thyroid_stimulating_hormone_(tsh)': 'thyroid_stimulating_hormone_(tsh)',
    'crp_high_sens': 'crp_high_sens',
    'procalcitonin': 'procalcitonin',
    'erythrocyte_sedimentation_rate_(esr)': 'erythrocyte_sedimentation_rate_(esr)',
    
    'icu_type': 'icu_type',
    'elapsed_icu': 'elapsed_icu_los',
    'elapsed_hosp': 'elapsed_hosp_los',
    'imc': 'imc',
    'ed': 'ed',
    'mtp': 'mtp',
    'c_diff': 'c_diff',
    'covid': 'covid',

    'infection': 'infection',
    'sepsis': 'sepsis',
    'on_dialysis': 'on_dialysis',
    'vent_status': 'on_vent',
    'on_pressors': 'on_pressors',

    'norepinephrine': 'norepinephrine',
    'epinephrine': 'epinephrine',       
    'dobutamine': 'dobutamine',
    'dopamine': 'dopamine',
    'phenylephrine': 'phenylephrine',
    'vasopressin': 'vasopressin',

    'norepinephrine_dose_unit': 'norepinephrine_dose_unit',
    'epinephrine_dose_unit': 'epinephrine_dose_unit',   
    'dobutamine_dose_unit': 'dobutamine_dose_unit',
    'dopamine_dose_unit': 'dopamine_dose_unit',
    'phenylephrine_dose_unit': 'phenylephrine_dose_unit',
    'vasopressin_dose_unit': 'vasopressin_dose_unit',
    'norepinephrine_dose_weight': 'norepinephrine_dose_weight',
    'epinephrine_dose_weight': 'epinephrine_dose_weight',
    'dobutamine_dose_weight': 'dobutamine_dose_weight',
    'dopamine_dose_weight': 'dopamine_dose_weight',
    'phenylephrine_dose_weight': 'phenylephrine_dose_weight',
    'vasopressin_dose_weight': 'vasopressin_dose_weight',
}

def prepare_df_for_csv(supertable: pd.DataFrame, features_map: dict) -> pd.DataFrame:
    """
    Prepare a supertable for csv export.
    """
    supertable_ = supertable.loc[:, SUPERTABLE_FEATURES_MAP.keys()]
    supertable_.rename(columns = SUPERTABLE_FEATURES_MAP, inplace = True)
    supertable_.index.name = "timestamp"
    
    return supertable_

def safe_read_pickle(path: str) -> pd.DataFrame:
    """
    Read a pickle file and return a dataframe.
    """
    try:
        df = pd.read_pickle(path)
    except Exception as e:
        print(f"Error reading pickle file {path}: {e}")
        return None
    return df

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process EMR data for a specific year')
    parser.add_argument('--year', type=int, help='The year for which data is being processed')
    parser.add_argument('--num_processes', type = int, default = 1, help = 'Total Number of Slurm Array processes')
    parser.add_argument('--processor_assignment', type = int, default = 0, help = 'Slurm Array number of this process')
    args = parser.parse_args()

    year = args.year
    num_processes = args.num_processes
    processor_assignment = args.processor_assignment

    supertable_root = Path("/labs/collab/K-lab-MODS/MODS-PHI/Encounter_Pickles/supertables2025/emory/supertables")
    supertable_path = supertable_root / str(year) / "Supertables"
    supertable_csv_path = supertable_root / str(year) / "CSVs"
    print(supertable_csv_path)
    supertable_csv_path.mkdir(exist_ok = True)

    files = list(supertable_path.glob("*.pkl"))
    print(f"Number of supertables found = {len(files)}")
    files = np.array(files)
    chunk_size = int( len(files) / num_processes)  
    print(f"The chunk size is {chunk_size}")
    
    # split the encounters into chunks
    list_of_chunks = np.array_split(files, num_processes)
    print(f"The list of chunks has {len(list_of_chunks)} unique dataframes.")
    
    # uses processor assignment number to select correct chunk
    process_list = list_of_chunks[processor_assignment]
    print(f"This process list has {len(process_list)} files")

    for i in range(len(process_list)):
        supertable = safe_read_pickle(process_list[i])
        supertable_csv = prepare_df_for_csv(supertable, SUPERTABLE_FEATURES_MAP)
        csn = process_list[i].stem 
        supertable_csv.to_csv(supertable_csv_path/ (csn + '.csv'))
        print("Saved csv to : ", supertable_csv_path / (csn + '.csv'))

if __name__ == "__main__":
    main()