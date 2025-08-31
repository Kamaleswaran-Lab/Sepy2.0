import pandas as pd
import os

# Base paths
base_dir = "/hpc/group/kamaleswaranlab/mimic_iv/builtdata/csv_exports"
output_dir = "/hpc/dctrl/yy450/Sepy2.0/0_mimic_gen_flat_files/mimic_flat_files"

# Load ICU input events (main infusion meds source)
inputevents = pd.read_csv(os.path.join(base_dir, "icu_inputevents.csv"))

# Standardize column names to pipeline requirements
infusion_final = pd.DataFrame({
    "csn": inputevents["hadm_id"],   # encounter id
    "pat_id": inputevents["subject_id"],
    "medication_id": inputevents["itemid"],  # will map via grouping file later
    "med_order_time": inputevents["starttime"],  # approximate as starttime
    "med_action_time": inputevents["endtime"],   # approximate as endtime
    "med_start": inputevents["starttime"],
    "med_stop": inputevents["endtime"],
    "med_order_route": "IV",  # Assume that all inputevents are IV
    "med_action_dose": inputevents["amount"],  # amount infused
    "med_action_dose_unit": inputevents["amountuom"]  # units
})

# Drop rows with missing essential identifiers
infusion_final = infusion_final.dropna(subset=["csn", "pat_id", "medication_id"])

# Save output
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, "INFUSIONMEDS.csv")
infusion_final.to_csv(out_path, index=False)

print("✅ INFUSIONMEDS.csv generated:", out_path, "shape:", infusion_final.shape)
print(infusion_final.head())
