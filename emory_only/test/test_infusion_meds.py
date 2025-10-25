import pandas as pd
import os
import sys
import pickle
from pathlib import Path
import numpy as np
sys.path.append("../")

import utils
import data_preprocess.medication_processor as medp

data_path = Path("/labs/collab/K-lab-MODS/MODS-PHI/")
emory_path = data_path / "Emory_Data"
fluid_path = Path("/labs/collab/K-lab-MODS/MODS-PHI/Fluids")
supertable_path = Path("/labs/collab/K-lab-MODS/MODS-PHI/Encounter_Pickles/emHolder_OutlierCorrected/2019_csvs")

allmeds = pd.read_csv("../scratch_files/allmeds_debug.csv")
encounters_list = allmeds["csn"].unique()

idx = 80
csn = 58426479062 #encounters_list[idx]


supertable = pd.read_csv(supertable_path / f"{csn}.csv")
supertable_index = supertable["Unnamed: 0"]
supertable_index = pd.to_datetime(supertable_index)
supertable.daily_weight_kg = supertable["daily_weight_kg"].ffill().bfill()
supertable = supertable.set_index(supertable_index)

meds = allmeds.loc[allmeds.csn == int(csn)]
meds = meds.sort_values("med_action_time")
imeds = meds.loc[meds["is_infusion"]]
print(f"Initial infusion meds: {imeds.shape}")
imeds = medp.impute_by_closest_location(imeds)
imeds = medp.process_premix(imeds)
print(f"After filtering: {imeds.shape}")

unique_order_ids = imeds["order_med_id"].unique()
print(f"{len(unique_order_ids)} unique order ids")
medsdict = {}

for order_id in unique_order_ids:

    print(f'\nProcessing order id: {order_id}')
    if order_id == str(8775327061):
        import pdb; pdb.set_trace()
    
    # Initialize medsdict for this order
    unique_meds = imeds.loc[imeds["order_med_id"] == order_id]['formulary_name'].unique()
    medsdict[order_id] = {}
    for med in unique_meds:
        medsdict[order_id][med] = {
            'rate': [], 'duration': [], 'med_start': [], 'med_stop': [], 'volume': [], 'set': False,
            'original_rate': None, 'original_volume': None, 'original_duration': None
        }
    
    # Get all rows for this order
    order_rows = imeds.loc[imeds["order_med_id"] == order_id].sort_values('med_action_time')
    all_processed_indices = set()
    
    # Process each medication in this order
    #import ipdb; ipdb.set_trace()
    for med_name in unique_meds:
        print(f"\n--- Processing medication: {med_name} ---")
        medsdict, med_processed_indices = medp.process_medication_timeline_new(
            order_id, med_name, order_rows, supertable, medsdict
        )
        all_processed_indices.update(med_processed_indices)
    
    print(f"Processed {len(all_processed_indices)} rows for order {order_id}")
    print("----------------------------------------------------")
    print(medsdict[order_id])

medsdf = medp.make_medsdict_to_dataframe(supertable, medsdict)
medsdf.to_csv(f"../scratch_files/medsdict_debug_{csn}.csv", index=False)
