import pandas as pd
import os

# Base directories
base_dir = "/hpc/group/kamaleswaranlab/mimic_iv/builtdata/csv_exports"
output_dir = "/hpc/home/yy450/link_kamaleswaranlab/mimic_iv/mimic_flat_files"
os.makedirs(output_dir, exist_ok=True)

# --- Load source files ---
patients = pd.read_csv(os.path.join(base_dir, "hosp_patients.csv"))
admissions = pd.read_csv(os.path.join(base_dir, "hosp_admissions.csv"))

# --- Step 1: Keep only admitted patients ---
admitted_ids = admissions["subject_id"].unique()
patients = patients[patients["subject_id"].isin(admitted_ids)]

# --- Step 2: Standardize gender ---
gender_map = {"M": "Male", "F": "Female"}
patients["gender"] = patients["gender"].map(gender_map)

# --- Step 3: Select representative race per patient ---
# If multiple admissions exist, take the first non-null race
race_map = (
    admissions[["subject_id", "race"]]
    .dropna(subset=["race"])
    .drop_duplicates(subset=["subject_id"], keep="first")
)

# --- Step 4: Merge patients with race ---
demo = patients.merge(race_map, on="subject_id", how="left")

# --- Step 5: Build DEMOGRAPHICS table ---
demo_final = pd.DataFrame({
    "pat_id": demo["subject_id"],
    "gender": demo["gender"],
    "race_code": demo["race"],        # directly use original race string
    "ethnicity_code": ""              # leave empty
})

# --- Step 6: Save output ---
out_path = os.path.join(output_dir, "DEMOGRAPHICS.csv")
demo_final.to_csv(out_path, index=False)

print("✅ DEMOGRAPHICS.csv generated:", out_path, "shape:", demo_final.shape)
print(demo_final.head())
