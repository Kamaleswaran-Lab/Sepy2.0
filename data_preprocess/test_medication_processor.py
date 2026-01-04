import pandas as pd
import os
from pathlib import Path
import hashlib
import sys
sys.path.append("../")
import data_preprocess.medication_processor as mp

def hash_value(value, hash_key = '123'):
    return hashlib.sha256((str(value) + hash_key).encode()).hexdigest()


meds = pd.read_csv("check_meds_1.csv")

super_table_time_index = pd.date_range(
            meds['med_action_time'].iloc[0],
            meds['med_action_time'].iloc[-1],
            freq='60min'
        )

placeholder_supertable = pd.DataFrame(index = super_table_time_index,  columns = ['daily_weight_kg'])
placeholder_supertable['daily_weight_kg'] = 75

meds = meds.sort_values("med_action_time")

# Filter to infusion medications only (i.e., not injections or syringes)
imeds = meds.loc[meds["is_infusion"]]
print(f"Initial infusion meds: {imeds.shape}")

# Process premix diluents (as these don't give us any information about the infusion)
imeds = mp.process_premix(imeds)
imeds = imeds.loc[imeds.formulary_name != "Not Recorded"]
print(f"After filtering: {imeds.shape}")

# Get unique order ids for this encounter
unique_order_ids = imeds["order_med_id"].unique()
print(f"{len(unique_order_ids)} unique order ids")
medsdict = {}

for idx, order_id in enumerate(unique_order_ids):
    print(f'\nProcessing order id: {order_id}')
    if idx == 4:
        import pdb; pdb.set_trace()

    # Get all rows for this order
    order_rows = imeds.loc[imeds["order_med_id"] == order_id].sort_values('med_action_time')
    unique_meds = order_rows['med_name'].unique()
    
    # Check if this is a dilution pair (exactly 2 medications) 
    # TODO: Add logic to handle more than 2 medications
    if len(unique_meds) == 2:
        print(f"Detected potential dilution pair with medications: {unique_meds}")
        
        fluid_meds = []
        non_fluid_meds = []
        
        # Check if any row for this medication has is_fluid=True
        for med_name in unique_meds:
            med_rows = order_rows[order_rows['med_name'] == med_name]
            if med_rows['is_fluid'].any():
                fluid_meds.append(med_name)
            else:
                non_fluid_meds.append(med_name)
        
        if len(fluid_meds) == 1 and len(non_fluid_meds) == 1:
            print(f"Confirmed dilution pair: {non_fluid_meds[0]} (medication) + {fluid_meds[0]} (fluid)")
            
            # Process as dilution pair
            dilution_result = process_dilution_pairs(
                order_id, fluid_meds[0], non_fluid_meds[0], order_rows, supertable
            )
            
            # Update medsdict with the result of the dilution pair processing
            if dilution_result:
                medsdict.update(dilution_result)
                print(f"Successfully processed dilution pair for order {order_id}")
            else:
                print(f"ERROR: Failed to process dilution pair for order {order_id} - skipping entire order")
            
            continue  # Skip individual processing for this order
        else:
            # Not a valid dilution pair - process individually
            print(f"Not a valid dilution pair (fluid_meds: {len(fluid_meds)}, non_fluid_meds: {len(non_fluid_meds)}) - processing individually")
    
    # Process individually (original logic)
    print(f"Processing {len(unique_meds)} medications individually")
    
    # Initialize medsdict for this order
    medsdict[order_id] = {}
    for med in unique_meds:
        medsdict[order_id][med] = {
            'rate': [], 'duration': [], 'med_start': [], 'med_stop': [], 'volume': [], 'set': False,
            'original_rate': None, 'original_volume': None, 'original_duration': None
        }
    
    all_processed_indices = set()
    
    # Process each medication in this order
    for med_name in unique_meds:
        print(f"\n--- Processing medication: {med_name} ---")
        medsdict, med_processed_indices = mp.process_medication_timeline_new(
            order_id, med_name, order_rows, placeholder_supertable, medsdict
        )
        all_processed_indices.update(med_processed_indices)
    
    print(f"Processed {len(all_processed_indices)} rows for order {order_id}")