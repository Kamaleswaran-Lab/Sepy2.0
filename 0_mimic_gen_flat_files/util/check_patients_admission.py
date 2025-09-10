import pandas as pd
import os

# Base paths
base_dir = "/hpc/group/kamaleswaranlab/mimic_iv/builtdata/csv_exports"

# Load patients and admissions
patients = pd.read_csv(os.path.join(base_dir, "hosp_patients.csv"))
admissions = pd.read_csv(os.path.join(base_dir, "hosp_admissions.csv"))

# Extract IDs
patient_ids = set(patients["subject_id"].unique())
admission_ids = set(admissions["subject_id"].unique())

# Find patients not admitted
not_in_admissions = patient_ids - admission_ids

print(f"Total patients: {len(patient_ids)}")
print(f"Patients with admissions: {len(admission_ids)}")
print(f"Patients without admissions: {len(not_in_admissions)}")

# Save list of patients without admissions (for inspection)
out_path = os.path.join(base_dir, "patients_without_admissions.csv")
pd.DataFrame({"subject_id": list(not_in_admissions)}).to_csv(out_path, index=False)

print(f"✅ Saved patients without admissions to {out_path}")
