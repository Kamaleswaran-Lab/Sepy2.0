import pandas as pd

def merge_o2_flow_and_vent_data(df_o2_flow: pd.DataFrame, 
                                o2_result_col: str, 
                                o2_units_col: str, 
                                o2_recorded_time_col : str,
                                o2_csn_col: str,
                                o2_patient_id_col: str,
                                df_vent: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the o2 flow and vent data s expected by SepyImport.

    Args:
        df_o2_flow: The o2 flow data.
        df_vent: The vent data.
        o2_result_col: The column name of the o2 result.
        o2_units_col: The column name of the o2 units.
        o2_recorded_time_col: The column name of the o2 recorded time.
        o2_csn_col: The column name of the o2 csn.
        o2_patient_id_col: The column name of the o2 patient id.

    Returns:
        The merged dataframe.
    """
    df_o2_flow = df_o2_flow.rename(columns={o2_result_col: "oxygen_flow_rate",
                                            o2_units_col: "oxygen_flow_rate_units",
                                            o2_recorded_time_col: "recorded_time",
                                            o2_csn_col: "csn",
                                            o2_patient_id_col: "pat_id"})
    
    drop_cols = [x for x in df_o2_flow.columns if x not in ["oxygen_flow_rate", "oxygen_flow_rate_units", "recorded_time", "csn", "pat_id"]]
    df_o2_flow = df_o2_flow.drop(columns=drop_cols)

    #Add df_o2_flow rows to df_vent, with nans in the columns that not in df_o2_flow
    df_vent = pd.concat([df_vent, df_o2_flow], ignore_index=True)

    return df_vent

#Function to convert icd9 column to icd10 codes in a dataframe according to the mapping in the icd10toicd9gem file 
def convert_icd9_to_icd10(df: pd.DataFrame, icd9_col: str, icd10_col: str, mapping_file: str) -> pd.DataFrame:
    """
    Convert icd9 column to icd10 codes in a dataframe.
    """
    #Read the mapping file
    mapping_df = pd.read_csv(mapping_file)

    #Create a dictionary of the mapping
    mapping = dict(zip(mapping_df["icd9cm"], mapping_df["icd10cm"]))

    #Apply the mapping to the dataframe
    df[icd10_col] = df[icd9_col].map(mapping)
    
    return df

