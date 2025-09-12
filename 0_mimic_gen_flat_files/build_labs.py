import pandas as pd
import os

base_dir = "/hpc/group/kamaleswaranlab/mimic_iv/builtdata/csv_exports"
output_dir = "/hpc/home/yy450/link_kamaleswaranlab/mimic_iv/mimic_flat_files"
os.makedirs(output_dir, exist_ok=True)

labitems = pd.read_csv(os.path.join(base_dir, "hosp_d_labitems.csv"))

# Output file
out_path = os.path.join(output_dir, "LABS.csv")
header_written = False

chunksize = 1_000_000  # process 1M rows at a time
total_written = 0

for chunk in pd.read_csv(os.path.join(base_dir, "hosp_labevents.csv"), chunksize=chunksize):
    labs = chunk.merge(labitems, on="itemid", how="left")

    labs_final = pd.DataFrame({
        "csn": labs["hadm_id"],
        "pat_id": labs["subject_id"],
        "component_id": labs["itemid"],

        # lab result: use valuenum if available, otherwise use value
        "lab_result": labs["valuenum"].combine_first(labs["value"]),
        "lab_result_time": labs["charttime"],
        "collection_time": labs["charttime"],
        "result_status": "Final",
        "proc_cat_id": labs["itemid"],
        "proc_cat_name": labs["fluid"],
        "proc_code": labs["itemid"],
        "proc_desc": labs["category"],

        "component": labs["label"],
        "loinc_code": "" 
    })

    labs_final = labs_final.dropna(subset=["pat_id", "csn", "component_id"])

    # Append to CSV
    labs_final.to_csv(out_path, mode="a", index=False, header=not header_written)
    header_written = True

    total_written += len(labs_final)
    print(f"✅ Processed chunk, total rows written so far: {total_written}")
