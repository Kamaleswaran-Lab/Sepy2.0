import pandas as pd
import os
from pathlib import Path
import hashlib
import sys
sys.path.append("../")
import data_preprocess.med_processing_utils as mpu
import data_preprocess.parameter_extractor as pe
import data_preprocess.medication_processor_multi as mpm
import data_preprocess.medication_processor as mp

def hash_value(value, hash_key = '123'):
    return hashlib.sha256((str(value) + hash_key).encode()).hexdigest()


meds = pd.read_csv("../scratch/check_meds_0013d826efd44adcdc4ca4b021c335169d7f34f196d7d63d275784bed7c0c17c.csv")

super_table_time_index = pd.date_range(
            meds['med_action_time'].iloc[0],
            meds['med_action_time'].iloc[-1],
            freq='60min'
        )

placeholder_supertable = pd.DataFrame(index = super_table_time_index,  columns = ['daily_weight_kg'])
placeholder_supertable['daily_weight_kg'] = 75

meds = meds.sort_values("med_action_time")

# Process premix diluents (as these don't give us any information about the infusion)
imeds = mpu.process_premix(meds)
imeds = imeds.loc[imeds.formulary_name != "Not Recorded"]
print(f"After filtering: {imeds.shape}")

# Get unique order ids for this encounter
unique_order_ids = imeds["order_med_id"].unique()
print(f"{len(unique_order_ids)} unique order ids")
medsdict = {}
all_meds_dict = {}


## Testing
order_id = unique_order_ids[46]
supertable = placeholder_supertable
print(f'\nProcessing order id: {order_id}')
    
# Get all rows for this order
order_rows = imeds.loc[imeds["order_med_id"] == order_id].sort_values('med_action_time')
unique_meds = order_rows['med_name'].unique()

supertable = placeholder_supertable
import pdb; pdb.set_trace()
if len(unique_meds) == 1:
    # Single medication - check if it's truly an infusion
    print(f"Single medication: {unique_meds[0]}")
    
    # Check if all rows are "Not Recorded" and not is_infusion
    all_not_recorded = (order_rows['med_action'] == 'Not Recorded').all()
    not_is_infusion = not order_rows['is_infusion'].iloc[0]  # Check first row
    
    if all_not_recorded and not_is_infusion:
        # Not truly an infusion - add to all_meds_dict
        print(f"  All rows are 'Not Recorded' and not is_infusion - adding to all_meds_dict")
        for idx, row in order_rows.iterrows():
            all_meds_dict = mp.add_to_all_meds_dict(row, all_meds_dict)
    else:
        # Process as normal infusion
        medsdict[order_id] = {}
        med_name = unique_meds[0]
        medsdict[order_id][med_name] = {
            'rate': [], 'duration': [], 'med_start': [], 'med_stop': [], 'volume': [], 'set': False,
            'original_rate': None, 'original_volume': None, 'original_duration': None
        }
        
        medsdict, _ = mp.process_medication_timeline_new(
            order_id, med_name, order_rows, supertable, medsdict
        )

else:
    # Multiple medications - use new logic
    print(f"Multiple medications ({len(unique_meds)}): {list(unique_meds)}")
    
    order_medsdict, order_non_inf_dict = mpm.process_order_multi_med(
        order_id, order_rows, supertable, all_meds_dict
    )
    
    if order_medsdict:
        medsdict.update(order_medsdict)

    if order_non_inf_dict:
        all_meds_dict.update(order_non_inf_dict)