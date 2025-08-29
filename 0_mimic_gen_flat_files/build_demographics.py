import pandas as pd
import os

# Base paths
base_dir = "/hpc/group/kamaleswaranlab/mimic_iv/builtdata/csv_exports"
output_dir = "/hpc/dctrl/yy450/Sepy2.0/0_mimic_gen_flat_files/mimic_flat_files"

# Load patients (demographics like gender, anchor age) and admissions (race)
patients = pd.read_csv(os.path.join(base_dir, "hosp_patients.csv"))
admissions = pd.read_csv(os.path.join(base_dir, "hosp_admissions.csv"))

# --- Gender mapping ---
gender_map = {"M": "Male", "F": "Female"}
patients["gender"] = patients["gender"].map(gender_map)

# --- Merge admissions (race info) with patients ---
# Take the first admission per patient for race (stable assignment)
admissions_race = admissions[["subject_id", "race"]].drop_duplicates("subject_id")
patients = patients.merge(admissions_race, on="subject_id", how="left")

# --- Race mapping (binned into categories) ---
def map_race_to_code(race: str) -> int:
    if pd.isna(race):
        return 309322
    race = race.upper()
    if "WHITE" in race:
        return 309315
    elif "BLACK" in race:
        return 309316
    elif "ASIAN" in race:
        return 309317
    elif "HISPANIC" in race:
        return 309318
    elif "AMERICAN INDIAN" in race:
        return 309319
    elif "NATIVE HAWAIIAN" in race or "PACIFIC ISLANDER" in race:
        return 309320
    elif "MULTIPLE" in race:
        return 309321
    elif race in ["OTHER", "PORTUGUESE", "SOUTH AMERICAN"]:
        return 309321
    elif race in ["PATIENT DECLINED TO ANSWER", "UNABLE TO OBTAIN", "UNKNOWN"]:
        return 309322
    else:
        return 309322  # default to unknown

patients["race_code"] = patients["race"].apply(map_race_to_code)

# --- Ethnicity mapping ---
def map_ethnicity(race: str) -> int:
    if pd.isna(race):
        return 312508
    if "HISPANIC" in race.upper():
        return 312507
    return 312508

patients["ethnicity_code"] = patients["race"].apply(map_ethnicity)

# --- Final DataFrame ---
demographics_final = pd.DataFrame({
    "pat_id": patients["subject_id"],
    "gender": patients["gender"],
    "race_code": patients["race_code"].astype(int),
    "ethnicity_code": patients["ethnicity_code"].astype(int)
})

# --- Save ---
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, "DEMOGRAPHICS.csv")
demographics_final.to_csv(out_path, index=False)

print("✅ DEMOGRAPHICS.csv generated:", out_path, "shape:", demographics_final.shape)
print(demographics_final.head())
