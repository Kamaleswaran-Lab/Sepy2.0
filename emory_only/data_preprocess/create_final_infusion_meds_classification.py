import pandas as pd 
import os
from pathlib import Path 

GROUPINGS = Path("~/Sepy2.0/groupings/")
volumes_file = GROUPINGS / "em_infusion_meds_volume_amounts.csv"
name_mapping_file = GROUPINGS / "em_infusion_meds_formulary_name_mapping.csv"

volumes_df = pd.read_csv(volumes_file)
volumes_df = volumes_df.drop(columns = ["med_name_generic"])
unnamed_cols = [col for col in volumes_df.columns if "Unnamed" in col]
volumes_df = volumes_df.drop(columns = unnamed_cols)

name_mapping_df = pd.read_csv(name_mapping_file)
unnamed_cols = [col for col in name_mapping_df.columns if "Unnamed" in col]
name_mapping_df = name_mapping_df.drop(columns = unnamed_cols)

volumes_df = volumes_df.merge(name_mapping_df, on = "formulary_name", how = "left")
volumes_df = volumes_df.drop(columns = ["is_fluids"])
print(volumes_df.shape)
volumes_df = volumes_df.drop_duplicates() 
print(volumes_df.shape)

classify_meds_file = GROUPINGS / "classified_medications_Aug19.csv"
classify_meds_df = pd.read_csv(classify_meds_file)
unnamed_cols = [col for col in classify_meds_df.columns if "Unnamed" in col]
classify_meds_df = classify_meds_df.drop(columns = unnamed_cols)
print(classify_meds_df.shape)

classify_meds_df = classify_meds_df.merge(volumes_df, on = "med_name", how = "right")
print(classify_meds_df.shape) 
classify_meds_df.to_csv(GROUPINGS / "em_infusion_meds_classification_final.csv", index = False)
