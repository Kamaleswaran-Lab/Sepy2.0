import pandas as pd
import os

# Define the base directory for MIMIC-IV CSV exports
base_dir = "/hpc/group/kamaleswaranlab/mimic_iv/builtdata/csv_exports"
output_dir = "/hpc/dctrl/yy450/Sepy2.0/0_mimic_gen_flat_files/mimic_flat_files"

# Load required source files
admissions = pd.read_csv(os.path.join(base_dir, "hosp_admissions.csv"))
patients = pd.read_csv(os.path.join(base_dir, "hosp_patients.csv"))
icustays = pd.read_csv(os.path.join(base_dir, "icu_icustays.csv"))

# --- Ensure proper data types ---
# Convert admissions times
admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")
admissions["edregtime"] = pd.to_datetime(admissions["edregtime"], errors="coerce")
admissions["edouttime"] = pd.to_datetime(admissions["edouttime"], errors="coerce")

# Convert ICU stay times and LOS
icustays["intime"] = pd.to_datetime(icustays["intime"], errors="coerce")
icustays["outtime"] = pd.to_datetime(icustays["outtime"], errors="coerce")
icustays["los"] = pd.to_numeric(icustays["los"], errors="coerce")

# --- Step 1: Merge admissions with patient demographics ---
encounter = admissions.merge(patients, on="subject_id", how="left")

# --- Step 2: Aggregate ICU stays information per hospital admission ---
icu_summary = (
    icustays.groupby("hadm_id")
    .agg(
        total_icu_days=("los", "sum"),
        first_icu_intime=("intime", "min"),
        last_icu_outtime=("outtime", "max"),
    )
    .reset_index()
)

# Join ICU summary back to admissions
encounter = encounter.merge(icu_summary, on="hadm_id", how="left")

# Fill missing ICU days with 0
encounter["total_icu_days"] = encounter["total_icu_days"].fillna(0)

# --- Step 3: Rename and align columns to match Emory ENCOUNTER specification ---
encounter_final = pd.DataFrame({
    "csn": encounter["hadm_id"],  # use hadm_id as encounter identifier
    "pat_id": encounter["subject_id"],
    "hospital_admission_date_time": encounter["admittime"],
    "hospital_discharge_date_time": encounter["dischtime"],
    "ed_presentation_time": encounter["edregtime"],  # ED registration time if available
    "encounter_type": "IN",  # all MIMIC patients are inpatients
    "age": encounter["anchor_age"],
    "discharge_to": encounter["discharge_location"],
    "pre_admit_location": encounter["admission_location"],
    "total_icu_days": encounter["total_icu_days"],
    "admit_reason": encounter["admission_type"],  # PLACEHOLDER: preserve original admission_type
})

# --- Step 4: Save the final ENCOUNTER file ---
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, "ENCOUNTER.csv")
encounter_final.to_csv(out_path, index=False)

print("✅ ENCOUNTER.csv generated:", out_path, "shape:", encounter_final.shape)
print(encounter_final.head())
