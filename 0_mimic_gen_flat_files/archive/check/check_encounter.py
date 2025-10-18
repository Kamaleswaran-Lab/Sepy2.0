import pandas as pd
import os

# Paths
base_dir = "/hpc/group/kamaleswaranlab/mimic_iv/builtdata"
export_dir = os.path.join(base_dir, "csv_exports")
output_dir = "/hpc/home/yy450/link_kamaleswaranlab/mimic_iv/mimic_flat_files"

admissions_path = os.path.join(export_dir, "hosp_admissions.csv")
encounter_path = os.path.join(output_dir, "ENCOUNTER.csv")

# --- Load files ---
admissions = pd.read_csv(admissions_path, usecols=["hadm_id"])
encounter = pd.read_csv(encounter_path, usecols=["csn"])

# Convert IDs to int (some may be float due to NA handling)
admissions["hadm_id"] = pd.to_numeric(admissions["hadm_id"], errors="coerce").astype("Int64")
encounter["csn"] = pd.to_numeric(encounter["csn"], errors="coerce").astype("Int64")

# --- Check counts ---
print("📊 Admissions count:", admissions["hadm_id"].nunique())
print("📊 Encounter count:", encounter["csn"].nunique())

# --- Find missing in Encounter ---
missing_in_encounter = set(admissions["hadm_id"]) - set(encounter["csn"])
print(f"❌ Missing in ENCOUNTER: {len(missing_in_encounter)}")

# --- Find extra in Encounter ---
extra_in_encounter = set(encounter["csn"]) - set(admissions["hadm_id"])
print(f"❌ Extra in ENCOUNTER: {len(extra_in_encounter)}")

# --- Optionally save lists for review ---
if missing_in_encounter:
    missing_path = os.path.join(output_dir, "missing_in_encounter.csv")
    pd.Series(list(missing_in_encounter), name="hadm_id").to_csv(missing_path, index=False)
    print(f"🔎 Saved missing hadm_id list to {missing_path}")

if extra_in_encounter:
    extra_path = os.path.join(output_dir, "extra_in_encounter.csv")
    pd.Series(list(extra_in_encounter), name="csn").to_csv(extra_path, index=False)
    print(f"🔎 Saved extra csn list to {extra_path}")
