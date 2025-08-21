import pandas as pd 
import os
from pathlib import Path 

GROUPINGS = Path("~/Sepy2.0/groupings/")
volumes_file = GROUPINGS / "em_infusion_meds_volume_amounts.csv"
name_mapping_file = GROUPINGS / "em_infusion_meds_formulary_name_mapping.csv"

volumes_df = pd.read_csv(volumes_file)
name_mapping_df = pd.read_csv(name_mapping_file)

volumes_df = volumes_df.merge(name_mapping_df, on = "formulary_name", how = "left")

classify_meds_file = GROUPINGS / "classified_medications_Aug19.csv"
classify_meds_df = pd.read_csv(classify_meds_file)

classify_meds_df = classify_meds_df.merge(volumes_df, on = "med_name", how = "left")

classify_meds_df.to_csv(GROUPINGS / "em_infusion_meds_classification_final.csv", index = False)
