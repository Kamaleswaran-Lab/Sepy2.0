import pandas as pd
import numpy as np
import sys
from pathlib import Path 
import os
import hashlib 

def hash_value(value, hash_key = '123'):
    return hashlib.sha256((str(value) + hash_key).encode()).hexdigest()

sys.path.append("../")

eroot= Path("/data/irb/surgery/pro00114885/EmoryDataset/noPHI")
emory_data = eroot 
fluid_path = eroot 

cjout = pd.read_csv(fluid_path / "CJSEPSIS_OUT_EO3.csv")
cjmeds = pd.read_csv(fluid_path / "CJSEPSIS_ORDEREDMEDS.dsv", sep = "|")

print(cjout.columns)
print(cjmeds.columns)
cjmeds["ORDER_PARENT_ID"] = cjmeds["ORDER_PARENT_ID"].astype(str)
cjmeds["ORDER_DT"] = pd.to_datetime(cjmeds["ORDER_DT"])
cjmeds["ENCOUNTER_NBR"] = cjmeds["ENCOUNTER_NBR"].astype(str)

cjout["order_ts"] = pd.to_datetime(cjout["order_ts"])
cjout["service_ts"] = pd.to_datetime(cjout["service_ts"])
cjout["csn"] = cjout["csn"].astype(str)

for year in range(2015, 2022):
    print("Processing year: ", year)
    if year == 2015:
        infusion_meds = pd.read_csv(emory_data / str(year) / f"CJSEPSIS_INFUSIONMEDS_{year}.dsv")
    else:
        infusion_meds = pd.read_csv(emory_data / str(year) / f"CJSEPSIS_INFUSIONMEDS_{year}.dsv", sep = "|")

    infusion_meds["order_med_id"] = infusion_meds["order_med_id"].astype(str)
    infusion_meds["order_med_id_hashed"] = infusion_meds["order_med_id"].apply(hash_value)
    infusion_meds["med_order_time"] = pd.to_datetime(infusion_meds["med_order_time"])
    infusion_meds["csn"] = infusion_meds["csn"].astype(str)
    infusion_meds["med_action_time"] = pd.to_datetime(infusion_meds["med_action_time"])
    print(infusion_meds.shape)

    allmeds = pd.merge(
        infusion_meds, cjmeds, 
        left_on=["csn", "order_med_id_hashed", "med_order_time", "med_name"], 
        right_on=["ENCOUNTER_NBR", "ORDER_PARENT_ID", "ORDER_DT", "ORDER_CATALOG_DESC"], 
        how="left"
    )

    print(allmeds.shape) 

    fluids_matched = pd.merge(
            allmeds, 
            cjout, 
            left_on=["csn", "med_name", "med_order_time", "med_action_time"], 
            right_on=["csn", "order_catalog_desc", "order_ts", "service_ts"], 
            how="left",
            suffixes = ["", "_fluids"]
        )

    print(fluids_matched.shape) 

    fluids_matched.to_csv(emory_data / str(year) / f'FLUIDS_MATCHED_{year}.dsv', sep = "|") 
    print("file saved to: ", emory_data / str(year) / f'FLUIDS_MATCHED_{year}.dsv')
    del fluids_matched
    del allmeds
    del infusion_meds

