import pandas as pd
import os

# Base directory
base_dir = "/hpc/home/yy450/link_kamaleswaranlab/mimic_iv/mimic_flat_files"

# Load files
demo = pd.read_csv(os.path.join(base_dir, "DEMOGRAPHICS.csv"))
enc = pd.read_csv(os.path.join(base_dir, "ENCOUNTER.csv"))

# Extract pat_id sets
demo_ids = set(demo["pat_id"].unique())
enc_ids = set(enc["pat_id"].unique())

# Compare sets
only_in_demo = demo_ids - enc_ids
only_in_enc = enc_ids - demo_ids

print(f"📊 DEMOGRAPHICS patients: {len(demo_ids)}")
print(f"📊 ENCOUNTER patients:    {len(enc_ids)}")

if not only_in_demo and not only_in_enc:
    print("✅ All pat_id match between DEMOGRAPHICS and ENCOUNTER!")
else:
    print(f"⚠️ Patients in DEMOGRAPHICS but not ENCOUNTER: {len(only_in_demo)}")
    print(f"⚠️ Patients in ENCOUNTER but not DEMOGRAPHICS: {len(only_in_enc)}")

    # Save mismatches
    out_demo = os.path.join(base_dir, "patients_only_in_demo.csv")
    out_enc = os.path.join(base_dir, "patients_only_in_enc.csv")
    pd.DataFrame({"pat_id": list(only_in_demo)}).to_csv(out_demo, index=False)
    pd.DataFrame({"pat_id": list(only_in_enc)}).to_csv(out_enc, index=False)

    print(f"🔎 Saved mismatch lists:\n - {out_demo}\n - {out_enc}")
