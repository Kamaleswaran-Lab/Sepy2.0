import pandas as pd
import os
from pathlib import Path
import hashlib
import numpy as np
import sys
sys.path.append("../")
import data_preprocess.medication_processor as mp

eroot = Path("/data/irb/surgery/pro00114885/EmoryDataset/noPHI") 

#Read the merged fluids file (infusion_meds flatfile, ORDERED_MEDS, and EO3)
year = 2019
infusiondf = pd.read_csv(eroot/ str(year) / f"FLUIDS_MATCHED_{2019}.dsv", sep = "|")

#Merge with bed locations
beds = pd.read_csv(eroot/ str(year) / f"CJSEPSIS_BEDLOCATION_{year}.dsv", sep = "|")
bed_labels = pd.read_csv(os.path.expandvars("$HOME/Sepy2.0/groupings/em_bed_labels.csv"))
beds = beds.merge(
        bed_labels[['bed_units', 'icu_type']], 
        left_on='bed_unit', 
        right_on='bed_units', 
        how='left'
    )
beds["bed_location_start"] = pd.to_datetime(beds["bed_location_start"])
mask = beds["icu_type"] == "sicu BEFORE 1/18/2018; cticu ON OR AFTER 1/18/2018"
cutoff_date = pd.to_datetime("1986-05-11 22:13:20")

beds.loc[mask, "icu_type"] = np.where(
    beds.loc[mask, "bed_location_start"] < cutoff_date,
    "sicu",
    "cticu"
)

mask = beds["icu_type"] == "cticu BEFORE 1/18/2018; micu ON OR AFTER 1/18/2018"
beds.loc[mask, "icu_type"] = np.where(
    beds.loc[mask, "bed_location_start"] < cutoff_date,
    "cticu",
    "micu"
)

mask = beds["icu_type"] == "sicu BEFORE 1/18/2018"
beds.loc[mask, "icu_type"] = np.where(
    beds.loc[mask, "bed_location_start"] < cutoff_date,
    "sicu",
    "other"
)
fluids_matched = pd.merge(infusiondf, beds, on = "csn", how = "left")
fluids_matched["med_action_time"] = pd.to_datetime(fluids_matched["med_action_time"])
fluids_matched["bed_location_start"] = pd.to_datetime(fluids_matched["bed_location_start"])
fluids_matched["bed_location_end"] = pd.to_datetime(fluids_matched["bed_location_end"])
fluids_matched["correct_bed"] = (fluids_matched["med_action_time"] <= fluids_matched["bed_location_end"]) &  (fluids_matched["med_action_time"] >= fluids_matched["bed_location_start"])
fluids_matched = fluids_matched.loc[fluids_matched.correct_bed]

#Impute formulary names which are "Not Recorded"
fluids_matched_imputed = mp.impute_by_closest_location_vectorized(fluids_matched)

#Get concentration_default and rate_default
conc_mapping = pd.read_csv("../groupings/em_meds_with_no_conc.csv")
conc_mapping = conc_mapping.loc[conc_mapping.concentration.notna() | conc_mapping.rate_per_hour.notna()]
fluids_matched_imputed.loc[fluids_matched_imputed.formulary_name.isin(conc_mapping.formulary_name), "concentration_default"] = fluids_matched_imputed["formulary_name"].map(dict(zip(conc_mapping["formulary_name"], conc_mapping["concentration"])))
fluids_matched_imputed.loc[fluids_matched_imputed.formulary_name.isin(conc_mapping.formulary_name), "rate_default"] = fluids_matched_imputed["formulary_name"].map(dict(zip(conc_mapping["formulary_name"], conc_mapping["rate_per_hour"])))

#Get amount, volume, and rate from formulary name
meds_mapping = pd.read_csv("../groupings/em_infusion_meds_classification_final.csv")

fluids_matched_imputed["volume_inf"] = fluids_matched_imputed["formulary_name"].map(dict(zip(meds_mapping["formulary_name"], meds_mapping["volume_numeric"])))
fluids_matched_imputed["volume_inf_unit"] = fluids_matched_imputed["formulary_name"].map(dict(zip(meds_mapping["formulary_name"], meds_mapping["volume_unit"])))

fluids_matched_imputed["amount_inf"] = fluids_matched_imputed["formulary_name"].map(dict(zip(meds_mapping["formulary_name"], meds_mapping["amount_numeric"])))
fluids_matched_imputed["amount_inf_unit"] = fluids_matched_imputed["formulary_name"].map(dict(zip(meds_mapping["formulary_name"], meds_mapping["amount_unit"])))

fluids_matched_imputed["is_anesthesia"] = fluids_matched_imputed["formulary_name"].map(dict(zip(meds_mapping["formulary_name"], meds_mapping["is_anesthesia"])))
fluids_matched_imputed["is_infusion"] = fluids_matched_imputed["formulary_name"].map(dict(zip(meds_mapping["formulary_name"], meds_mapping["is_infusion"])))
fluids_matched_imputed["is_fluid"] = fluids_matched_imputed["formulary_name"].map(dict(zip(meds_mapping["formulary_name"], meds_mapping["is_fluids"])))

#Get med_class and med_subclass
fluids_matched_imputed["med_class"] = fluids_matched_imputed["med_name"].map(dict(zip(meds_mapping["med_name"], meds_mapping["med_class"])))
fluids_matched_imputed["med_subclass"] = fluids_matched_imputed["med_name"].map(dict(zip(meds_mapping["med_name"], meds_mapping["med_subclass"])))

# Step 3.7: Parse concentration_default and calculate volumes
unique_concentrations = fluids_matched_imputed.loc[
    fluids_matched_imputed.concentration_default.notna(), 
    'concentration_default'
].unique()

# Parse each unique concentration and convert amounts to milligrams
concentration_parsed_map = {}
for conc_str in unique_concentrations:
    parsed = mp.parse_concentration(conc_str)
    
    if parsed['amount'] is not None and parsed['amount_unit'] is not None:
        # Map the parsed amount_unit to standardized unit name
        amount_unit_standardized = mp.amount_unit_mapping.get(
            parsed['amount_unit'].lower(), 
            parsed['amount_unit']
        )
        
        # Convert amount to milligrams
        amount_in_mg = mp.convert_units(
            parsed['amount'], 
            amount_unit_standardized, 
            'Milligram'
        )
        
        if amount_in_mg is not None:
            # Volume stays the same - only amount unit changes
            # Concentration automatically adjusts: mg/mL instead of g/mL, etc.
            concentration_parsed_map[conc_str] = {
                'amount_mg': amount_in_mg,
                'volume_ml': parsed['volume_ml'],
                'concentration_mg_per_ml': amount_in_mg / parsed['volume_ml'] if parsed['volume_ml'] > 0 else None
            }
        else:
            # Could not convert - store original values
            concentration_parsed_map[conc_str] = {
                'amount_mg': None,
                'volume_ml': parsed['volume_ml'],
                'concentration_mg_per_ml': None
            }
    else:
        # Parsing failed
        concentration_parsed_map[conc_str] = {
            'amount_mg': None,
            'volume_ml': None,
            'concentration_mg_per_ml': None
        }

# Map parsed concentrations back to dataframe
fluids_matched_imputed['conc_amount_mg'] = fluids_matched_imputed['concentration_default'].map(
    lambda x: concentration_parsed_map.get(x, {}).get('amount_mg') if pd.notna(x) else None
)
fluids_matched_imputed['conc_volume_ml'] = fluids_matched_imputed['concentration_default'].map(
    lambda x: concentration_parsed_map.get(x, {}).get('volume_ml') if pd.notna(x) else None
)
fluids_matched_imputed['conc_mg_per_ml'] = fluids_matched_imputed['concentration_default'].map(
    lambda x: concentration_parsed_map.get(x, {}).get('concentration_mg_per_ml') if pd.notna(x) else None
)

# Apply the calculation only to rows with concentration_default
mask_has_concentration = fluids_matched_imputed['concentration_default'].notna()
fluids_matched_imputed.loc[mask_has_concentration, 'volume_from_concentration'] = (
    fluids_matched_imputed.loc[mask_has_concentration].apply(mp.calculate_volume_from_concentration, axis=1)
)

# Step 3.8: Parse rate_default to extract numeric values (always in ml/hr)
fluids_matched_imputed['rate_default_numeric'] = fluids_matched_imputed['rate_default'].apply(mp.parse_rate_default)

# Step 4 & 5: Drop NR rows and remove certain cohorts
not_recorded = fluids_matched_imputed.loc[fluids_matched_imputed.formulary_name == "Not Recorded"]
recorded = fluids_matched_imputed.loc[~(fluids_matched_imputed.formulary_name == "Not Recorded")]
not_recorded_in_recorded = not_recorded.loc[not_recorded.order_med_id.isin(recorded.order_med_id.unique())]
really_not_recorded = not_recorded.loc[~not_recorded.order_med_id.isin(recorded.order_med_id.unique())]

fluids_matched_imputed = fluids_matched_imputed.loc[~fluids_matched_imputed.order_med_id.isin(really_not_recorded.order_med_id)]
fluids_matched_imputed = fluids_matched_imputed.loc[~fluids_matched_imputed.bed_label.isin([
    'NURSERY INTENSIVE', 'REHAB 4-BED', 'NURSERY INTERMEDIATE', 'NURSERY LEVEL 2',
    'OBSERVATION','NURSERY'
])]

#Step 6: PROCESS ROW BY ROW
fluids_matched_imputed = fluids_matched_imputed.sort_values(by = ["csn", "med_action_time"])
csns = fluids_matched_imputed.csn.unique()

csn = csns[0]

###Per CSN:
meds = fluids_matched.loc[fluids_matched.csn == csn_]
#Read the supertable - rn im creating a placeholder
super_table_time_index = pd.date_range(
            meds['med_action_time'].iloc[0],
            meds['med_action_time'].iloc[-1],
            freq='60min'
        )
placeholder_supertable = pd.DataFrame(index = super_table_time_index,  columns = ['daily_weight_kg'])
placeholder_supertable['daily_weight_kg'] = 75

meds = meds.sort_values("med_action_time")

# Filter to infusion medications only (i.e., not injections or syringes)
print(f"Initial infusion meds: {meds.shape}")

# Process premix diluents (as these don't give us any information about the infusion)
imeds = mp.process_premix(meds)
print(f"After filtering premix: {imeds.shape}")

# Get unique order ids for this encounter
unique_order_ids = imeds["order_med_id"].unique()
print(f"{len(unique_order_ids)} unique order ids")
medsdict = {}
all_meds_dict = {}

for order_id in unique_order_ids:
    supertable = placeholder_supertable
    print(f'\nProcessing order id: {order_id}')
        
    # Get all rows for this order
    order_rows = imeds.loc[imeds["order_med_id"] == order_id].sort_values('med_action_time')
    unique_meds = order_rows['med_name'].unique()
    
    # Check is_infusion and is_fluid consistency for each unique medication
    med_properties = {}
    for med_name in unique_meds:
        med_rows = order_rows[order_rows['med_name'] == med_name]
        
        # Check is_infusion consistency
        is_infusion_values = med_rows['is_infusion'].dropna().unique()
        if len(is_infusion_values) > 1:
            print(f"ERROR: {med_name} has inconsistent is_infusion values: {is_infusion_values}")
            is_infusion = False  #By default False
        elif len(is_infusion_values) == 1:
            is_infusion = is_infusion_values[0]
        else:
            is_infusion = False  # Default if all are NaN
        
        # Check is_fluid consistency
        is_fluid_values = med_rows['is_fluid'].dropna().unique()
        if len(is_fluid_values) > 1:
            print(f"ERROR: {med_name} has inconsistent is_fluid values: {is_fluid_values}")
            is_fluid = is_fluid_values[0]  # Use first value but flag error
        elif len(is_fluid_values) == 1:
            is_fluid = is_fluid_values[0]
        else:
            is_fluid = False  # Default if all are NaN
        
        med_properties[med_name] = {
            'is_infusion': is_infusion,
            'is_fluid': is_fluid
        }
    
    # Step 1: Process all non-infusion medications first
    non_infusion_meds = [med for med in unique_meds if not med_properties[med]['is_infusion']]
    
    if non_infusion_meds:
        print(f"Processing {len(non_infusion_meds)} non-infusion medication(s): {non_infusion_meds}")
        for med_name in non_infusion_meds:
            non_infusion_rows = order_rows[order_rows['med_name'] == med_name]
            for idx, row in non_infusion_rows.iterrows():
                all_meds_dict = mp.add_to_all_meds_dict(row, all_meds_dict)
            print(f"Added non-infusion med {med_name} to all_meds_dict")
    
    # Step 2: Filter out non-infusion meds to get remaining infusion meds
    remaining_rows = order_rows[~order_rows['med_name'].isin(non_infusion_meds)]
    remaining_unique_meds = remaining_rows['med_name'].unique()
    
    print(f"Remaining infusion medications: {len(remaining_unique_meds)}")
    
    # Step 3: Check if all remaining rows are "Not Recorded"
    if len(remaining_rows) > 0 and (remaining_rows['formulary_name'] == 'Not Recorded').all():
        print("All remaining rows are 'Not Recorded' - skipping (likely diluents)")
    
    # Step 4: Process based on number of remaining unique medications
    elif len(remaining_unique_meds) == 0:
        print("No infusion medications to process")
    
    elif len(remaining_unique_meds) == 1:
        # Single infusion medication
        med_name = remaining_unique_meds[0]
        print(f"Processing 1 infusion medication: {med_name}")
        
        # Initialize medsdict for this order
        medsdict[order_id] = {}
        medsdict[order_id][med_name] = {
            'rate': [], 'duration': [], 'med_start': [], 'med_stop': [], 'volume': [], 'set': False,
            'original_rate': None, 'original_volume': None, 'original_duration': None
        }
        
        medsdict, med_processed_indices = mp.process_medication_timeline_new(
            order_id, med_name, remaining_rows, supertable, medsdict
        )
        print(f"Processed {len(med_processed_indices)} rows for {med_name}")
    
    elif len(remaining_unique_meds) == 2:
        # Two infusion medications - check for dilution pair
        med_name_1, med_name_2 = remaining_unique_meds[0], remaining_unique_meds[1]
        print(f"Processing 2 infusion medications: {med_name_1} and {med_name_2}")
        
        # Check if one is fluid and one is not (dilution pair criteria)
        if med_properties[med_name_1]['is_fluid'] != med_properties[med_name_2]['is_fluid']:
            fluid_med = med_name_1 if med_properties[med_name_1]['is_fluid'] else med_name_2
            non_fluid_med = med_name_2 if med_properties[med_name_1]['is_fluid'] else med_name_1
            
            print(f"Confirmed dilution pair: {non_fluid_med} (medication) + {fluid_med} (fluid)")
            
            # Process as dilution pair
            dilution_result = mp.process_dilution_pairs(
                order_id, fluid_med, non_fluid_med, remaining_rows, supertable
            )
            
            # Update medsdict with the result
            if dilution_result:
                medsdict.update(dilution_result)
                print(f"Successfully processed dilution pair for order {order_id}")
            else:
                print(f"ERROR: Failed to process dilution pair for order {order_id}")
        else:
            # Both infusions but not a valid dilution pair - process individually
            print(f"Both are infusions but not a valid dilution pair (both fluid={med_properties[med_name_1]['is_fluid']}) - processing individually")
            
            # Initialize medsdict for this order
            medsdict[order_id] = {}
            for med in remaining_unique_meds:
                medsdict[order_id][med] = {
                    'rate': [], 'duration': [], 'med_start': [], 'med_stop': [], 'volume': [], 'set': False,
                    'original_rate': None, 'original_volume': None, 'original_duration': None
                }
            
            all_processed_indices = set()
            
            # Process each medication individually
            for med_name in remaining_unique_meds:
                print(f"\n--- Processing medication: {med_name} ---")
                medsdict, med_processed_indices = mp.process_medication_timeline_new(
                    order_id, med_name, remaining_rows, supertable, medsdict
                )
                all_processed_indices.update(med_processed_indices)
    
    else:
        # More than 2 infusion medications
        print(f"ERROR: Order {order_id} has {len(remaining_unique_meds)} infusion medications after filtering - not yet implemented")
        raise NotImplementedError(f"Cannot handle {len(remaining_unique_meds)} infusion medications in one order")
