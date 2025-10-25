import pandas as pd
import numpy as np
import sys
from pathlib import Path 
import os
sys.path.append("../")


emory_data = Path("/labs/collab/K-lab-MODS/MODS-PHI/Emory_Data/")
fluid_path = Path("/labs/collab/K-lab-MODS/MODS-PHI/Fluids")

cjout = pd.read_csv(fluid_path / "CJSEPSIS_OUT_EO3.txt", sep = "|")
cjmeds = pd.read_csv(fluid_path / "CJSEPSIS_ORDEREDMEDS.csv")
cjmeds["ORDER_PARENT_ID"] = cjmeds["ORDER_PARENT_ID"].astype(str)
cjmeds["ORDER_DT"] = pd.to_datetime(cjmeds["ORDER_DT"])
cjmeds["ENCOUNTER_NBR"] = cjmeds["ENCOUNTER_NBR"].astype(int)

cjout["ORDER_TS"] = pd.to_datetime(cjout["ORDER_TS"])
cjout["SERVICE_TS"] = pd.to_datetime(cjout["SERVICE_TS"])
cjout["CSN"] = cjout["CSN"].astype(int)

for year in range(2015, 2022):
    print("Processing year: ", year)
    infusion_meds = pd.read_csv(emory_data / str(year) / f"CJSEPSIS_INFUSIONMEDS_{year}.dsv", sep = "|")

    infusion_meds["order_med_id"] = infusion_meds["order_med_id"].astype(str)
    infusion_meds["med_order_time"] = pd.to_datetime(infusion_meds["med_order_time"])
    infusion_meds["csn"] = infusion_meds["csn"].astype(int)
    infusion_meds["med_action_time"] = pd.to_datetime(infusion_meds["med_action_time"])
    print(infusion_meds.shape)

    allmeds = pd.merge(
        infusion_meds, cjmeds, 
        left_on=["csn", "order_med_id", "med_order_time", "med_name"], 
        right_on=["ENCOUNTER_NBR", "ORDER_PARENT_ID", "ORDER_DT", "ORDER_CATALOG_DESC"], 
        how="left"
    )

    print(allmeds.shape) 

    fluids_matched = pd.merge(
            allmeds, 
            cjout, 
            left_on=["csn", "med_name", "med_order_time", "med_action_time"], 
            right_on=["CSN", "ORDER_CATALOG_DESC", "ORDER_TS", "SERVICE_TS"], 
            how="left",
            suffixes = ["", "_fluids"]
        )

    print(fluids_matched.shape) 

    fluids_matched.to_csv(emory_data / str(year) / f'FLUIDS_MATCHED_{year}.dsv', sep = "|") 
    print("file saved to: ", emory_data / str(year) / f'FLUIDS_MATCHED_{year}.dsv')
    del fluids_matched
    del allmeds
    del infusion_meds

