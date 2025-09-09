import pandas as pd
import os

# Base directories
base_dir = "/hpc/group/kamaleswaranlab/mimic_iv/builtdata"
export_dir = os.path.join(base_dir, "csv_exports")
concepts_dir = os.path.join(base_dir, "csv_concepts_exports")
output_dir = "/hpc/home/yy450/link_kamaleswaranlab/mimic_iv/mimic_flat_files"
os.makedirs(output_dir, exist_ok=True)

# --- Load source files ---
admissions = pd.read_csv(os.path.join(export_dir, "hosp_admissions.csv"))
patients = pd.read_csv(os.path.join(export_dir, "hosp_patients.csv"))  # for anchor_age
icustay_detail = pd.read_csv(os.path.join(concepts_dir, "icustay_detail.csv"))

# --- Ensure proper datetime types ---
admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")
admissions["edregtime"] = pd.to_datetime(admissions["edregtime"], errors="coerce")

icustay_detail["icu_intime"] = pd.to_datetime(icustay_detail["icu_intime"], errors="coerce")
icustay_detail["icu_outtime"] = pd.to_datetime(icustay_detail["icu_outtime"], errors="coerce")
icustay_detail["los_icu"] = pd.to_numeric(icustay_detail["los_icu"], errors="coerce")

# --- Step 1: Aggregate ICU stay durations per hadm_id ---
icu_summary = (
    icustay_detail.groupby("hadm_id")
    .agg(
        total_icu_days=("los_icu", "sum"),
        first_icu_intime=("icu_intime", "min"),
        last_icu_outtime=("icu_outtime", "max"),
    )
    .reset_index()
)

# --- Step 2: Merge admissions (main table) with patients (for age) ---
encounter = admissions.merge(
    patients[["subject_id", "anchor_age"]],
    on="subject_id",
    how="left"
)

# --- Step 3: Merge ICU summary into admissions ---
encounter = encounter.merge(icu_summary, on="hadm_id", how="left")
encounter["total_icu_days"] = encounter["total_icu_days"].fillna(0)

# --- Step 4: Align columns to ENCOUNTER spec ---
encounter_final = pd.DataFrame({
    "csn": encounter["hadm_id"],
    "pat_id": encounter["subject_id"],
    "hospital_admission_date_time": encounter["admittime"],
    "hospital_discharge_date_time": encounter["dischtime"],
    "ed_presentation_time": encounter["edregtime"],
    "encounter_type": "IN",  # all MIMIC patients are inpatients
    "age": encounter["anchor_age"],  # use patients.anchor_age
    "discharge_to": encounter["discharge_location"],
    "pre_admit_location": encounter["admission_location"],
    "total_icu_days": encounter["total_icu_days"],
    "admit_reason": encounter["admission_type"],  # placeholder for Emory admit_reason
})

# --- Step 5: Save to CSV ---
out_path = os.path.join(output_dir, "ENCOUNTER.csv")
encounter_final.to_csv(out_path, index=False)

print("✅ ENCOUNTER.csv generated:", out_path, "shape:", encounter_final.shape)
print(encounter_final.head())
