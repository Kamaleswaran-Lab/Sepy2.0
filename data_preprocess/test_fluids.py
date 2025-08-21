import pandas as pd
import os
import sys
import pickle
from pathlib import Path
import numpy as np
sys.path.append("../")

import utils
import importlib
importlib.reload(utils)

def check_duration_in_desc(row):
   duration = row["DURATION_INFUSION_ORDERED_QTY"]
   desc = row["ORDER_CLINICAL_DESC"]
   
   if pd.isna(duration) or pd.isna(desc):
       return {"found": False, "unit": None}
   
   desc = desc.lower()
   
   # Check for hours - handle both integer and decimal
   if duration == int(duration):
       hour_pattern = f"{int(duration)} hr"
   else:
       hour_pattern = f"{duration} hr"
   
   if hour_pattern in desc:
       return {"found": True, "unit": "hours"}
   
   # Check for minutes - handle both integer and decimal
   if duration == int(duration):
       minute_pattern = f"{int(duration)} minute"
   else:
       minute_pattern = f"{duration} minute"
   
   if minute_pattern in desc:
       return {"found": True, "unit": "minutes"}
   
   return {"found": False, "unit": None}

def process_each_row(row, ongoing_infusion, medsdf, medsdict, meds, supertable, supertable_index):
    parent = False
    sus = False
    med_start = None
    med_stop = None
    rate_inf = None
    med_action = None
    
    weight = supertable.loc[supertable_index.row["med_action_time"]].values[0]
    print(weight)
    if (not row["order_med_id"] in ongoing_infusion) or ((row["order_med_id"] in ongoing_infusion) and (row["med_action"] != "Infuse")):
        
        if row["formulary_name"] not in medsdict:
            medsdict[row["formulary_name"]] = 1
            medsdf[row["formulary_name"]] = 0.0
        else:
            medsdict[row["formulary_name"]] += 1
            
        volume = False
        rate = False
        duration = False
        if row["ORDER_PARENT_ID"] and not pd.isna(row["ORDER_PARENT_ID"]) :
            parent = True
            duration_parent = row["DURATION_INFUSION_ORDERED_QTY"]
            volume_parent = row["VOLUME_ORDERED_QTY"]
            volume_unit = row["VOLUME_UNIT_MEASURE"]
            rate_parent = row["INFUSION_ORDER_RT"]
            rate_unit = row["RATE_UNIT_MEASURE"]
            
            if volume_unit != "Milliliter":
                sus = True
        
            if rate_unit != "Milliliter/hour":
                sus = True
                
            #check duration
            result = check_duration_in_desc(row)
            if result["found"]:
                if result["unit"] == "minutes":
                    duration_parent = duration_parent/60.0
            elif rate_parent and volume_parent and abs(duration_parent - rate_parent*volume_parent) > 10:
                sus = True
            print("Parent:", parent, sus)
            print("volume: ", volume_parent, volume_unit, " rate: ", rate_parent, rate_unit, " duration: ", duration_parent)
        
        deal = False
        clinical_desc = False
        if row["ORDER_CLINICAL_DESC"] and not pd.isna(row["ORDER_CLINICAL_DESC"]):
            clinical_desc = True
            params = utils.parse_clinical_description(row["ORDER_CLINICAL_DESC"])
            print(params["volume"])
            if params["volume"]:
                if len(params["volume"]) > 1:
                    if 'total_volume' in params['volume_unit']:
                        params['volume'] = params["volume"][params["volume_unit"].index('total_volume')]
                        params['volume_unit'] = 'total_volume'
                    else:
                        deal = True
                else:
                    params["volume"] = params["volume"][0]
                    params["volume_unit"] = params["volume_unit"][0]
        
                    
            if params["rate"]:
                if len(params["rate"]) > 1:
                    deal = True
                else:
                    params["rate"] = params["rate"][0]
            
            if params["duration"]:
                params["duration"] = list(set(params["duration"]))
                if len(params["duration"]) > 1 and 0.0 in params["duration"]:
                    params["duration"].remove(0.0)
                if len(params["duration"]) > 1:
                    if not deal:
                        duration = params["volume"]/params["rate"]
                        if duration in params["duration"]:
                            params["duration"] = duration
                        else:
                            deal = True
                else:
                    params["duration"] = params["duration"][0]
            
            print(row["ORDER_CLINICAL_DESC"])
            print(params)
        
            if not deal:
                volume_desc = params["volume"]
                rate_desc = params["rate"]
                duration_desc = params["duration"]
                if volume_desc and rate_desc and duration_desc and abs(duration_desc - (volume_desc/rate_desc)) > 10:
                    deal = True
            print("Clinical desc deal:", deal)
        
        final_check = True
        if not parent and (clinical_desc and not deal):
            volume = volume_desc
            rate = rate_desc
            duration = duration_desc
            print("Infusion params clear - from clinical desc")
        elif not parent and (clinical_desc and deal):
            print("Deal with this case - no parent and sus clinical desc")
            final_check = False
        elif (parent and not sus) and (not clinical_desc):
            print("infusion params clear - from parent order id")
            volume = volume_parent
            rate = rate_parent
            duration = duration_parent
        elif (parent and sus) and (not clinical_desc):
            print("Deal with this case - no clinical desc and sus parent params")
            final_check = False
        elif (parent and not sus) and (clinical_desc and not deal):
            volume = volume_parent
            rate = rate_parent
            duration = duration_parent
            print("using parent")
        elif (parent and not sus) and (clinical_desc and deal):
            print("Check - infusion params frm parents, but clinical desc unclear")
            
            if params["duration"] and not isinstance(params["duration"], float) and (len(params["duration"]) > 1):
                if duration_parent in params["duration"]:
                    duration = duration_parent
                else:
                    final_check = False
            elif isinstance(params["duration"], float):
                if params["duration"] == duration_parent:
                    duration = duration_parent
                else:
                    final_check = False
        
            if params["rate"] and not isinstance(params["rate"], float) and (len(params["rate"]) > 1):
                if rate_parent in params["rate"]:
                    rate = rate_parent
                else:
                    final_check = False
            elif isinstance(params["rate"], float):
                if params["rate"] == rate_parent:
                    rate = rate_parent
                else:
                    final_check = False
        
            if params["volume"] and not isinstance(params["volume"], float) and (len(params["volume"]) > 1):
                if volume_parent in params["volume"]:
                    volume = volume_parent
                else:
                    final_check = False
            elif isinstance(params["volume"], float):
                if params["volume"] == volume_parent:
                    volume = volume_parent
                else:
                    final_check = False
                    
            if not final_check:
                volume = volume_desc
                rate = rate_desc
                duration = duration_desc
        elif (parent and sus) and (clinical_desc and not deal):
            volume = volume_desc
            rate = rate_desc
            duration = duration_desc
            
        print("volume: ", volume, " rate: ", rate, " duration: ", duration, "final check passed: ", final_check )
        
        if not volume:
            params_2 = utils.extract_volume_detailed(row["formulary_name"])
            volume = params_2["raw_value"] if params_2["unit"] == "mL" else None
        
        if not pd.isna(row["med_start"]):
            med_start = pd.to_datetime(row["med_start"])
            print("Start: ", med_start)
        
        if not pd.isna(row["med_stop"]):
            med_stop = pd.to_datetime(row["med_stop"])
            print("Stop: ", med_stop) 
        if not pd.isna(row["med_action_dose"]):
            med_action_dose = row["med_action_dose"]
            med_action_dose_unit = row["med_action_dose_unit"]
            weight = row["weight"]  # Assuming weight is available
            
            rate = None
            duration = None
            
            if med_action_dose_unit == 'Not Recorded':
                rate = None
                
            elif med_action_dose_unit == 'Milligrams/Minute':
                rate = med_action_dose * 60
                rate_unit = 'Milligrams/Hour'
                
            elif med_action_dose_unit == 'Micrograms/Hour':
                rate = med_action_dose
                rate_unit = 'Micrograms/Hour'
                
            elif med_action_dose_unit == 'Microgram/Kilogram/Minute':
                rate = med_action_dose * weight * 60
                rate_unit = 'Micrograms/Hour'
                
            elif med_action_dose_unit == 'Microgram/Kilogram/Hour':
                rate = med_action_dose * weight
                rate_unit = 'Micrograms/Hour'
                
            elif med_action_dose_unit == 'Milligram/Hour':
                rate = med_action_dose
                rate_unit = 'Milligrams/Hour'
                
            elif med_action_dose_unit == 'Milliequivalents/Minute':
                rate = med_action_dose * 60
                rate_unit = 'Milliequivalents/Hour'
                
            elif med_action_dose_unit == 'ng/kg/min':
                rate = med_action_dose * weight * 60
                rate_unit = 'Nanograms/Hour'
                
            elif med_action_dose_unit == 'Milligram/Kilogram/Hour':
                rate = med_action_dose * weight
                rate_unit = 'Milligrams/Hour'
                
            elif med_action_dose_unit == 'U/Hr':
                rate = med_action_dose
                rate_unit = 'Units/Hour'
                
            elif med_action_dose_unit == 'Micrograms/Minute':
                rate = med_action_dose * 60
                rate_unit = 'Micrograms/Hour'
                
            elif med_action_dose_unit == 'Unit/Minute':
                rate = med_action_dose * 60
                rate_unit = 'Units/Hour'
                
            elif med_action_dose_unit == 'Grams/Hour':
                rate = med_action_dose
                rate_unit = 'Grams/Hour'
                
            elif med_action_dose_unit == 'Unit/Kilogram/Hour':
                rate = med_action_dose * weight
                rate_unit = 'Units/Hour'
                
            elif med_action_dose_unit == 'Milligram/Kilogram/Minute':
                rate = med_action_dose * weight * 60
                rate_unit = 'Milligrams/Hour'
                
            elif med_action_dose_unit == 'Milliequivalents/Kilogram/Hour':
                rate = med_action_dose * weight
                rate_unit = 'Milliequivalents/Hour'
                
            elif med_action_dose_unit == 'Milligrams/Millilit':
                rate = med_action_dose
                rate_unit = 'Milligrams/Milliliter'  # Concentration, not rate
                
            elif med_action_dose_unit == 'minute(s)':
                duration = med_action_dose / 60.0  # Convert to hours
                
            elif med_action_dose_unit == 'Hour':
                duration = med_action_dose  # Already in hours
            
            if rate is not None:
                print(f"fn Rate: {rate} {rate_unit}")
            if duration is not None:
                print(f"fn Duration: {duration} hours")
        
        if not pd.isna(row["med_action"]):
            med_action = row["med_action"]
            print("Med action " , med_action)

        if med_action == "Begin Bag":
            meds_slice = meds.loc[(meds["order_med_id"] == row["order_med_id"]) & (meds["formulary_name"] == row["formulary_name"])]
            meds_slice = meds_slice.sort_values("med_action_time")
            
        if len(meds_slice) == 1:
            print("Only one instance recorded")
        else:
            ongoing_infusion[row["order_med_id"]] = row["med_action_time"]
            start_time = row["med_action_time"]
            
            # Get current row index
            current_idx = row.name
            
            # Find subsequent rows for this medication
            subsequent_rows = meds_slice[meds_slice.index > current_idx].sort_values('med_action_time')
            
            end_time = None
            for _, next_row in subsequent_rows.iterrows():
                if next_row["med_action"] == "Infuse":
                    # Keep tracking this as potential end time
                    last_infuse_time = next_row["med_action_time"]
                elif next_row["med_action"] == "Begin Bag":
                    # Stop here, use the last "Infuse" time
                    end_time = last_infuse_time
                    break
            
            if end_time:
                # Calculate duration to nearest whole hour
                raw_duration = pd.to_datetime(end_time) - pd.to_datetime(start_time)
                duration_hours = round(raw_duration.total_seconds() / 3600.0)
                duration = pd.Timedelta(hours=duration_hours)
                print(f"Duration: {duration_hours} hours")
                        
        if med_start and med_stop:
            med_start_dt = pd.to_datetime(med_start)
            med_stop_dt = pd.to_datetime(med_stop)
            duration_meds = med_stop_dt - med_start_dt
            duration_meds = duration_meds.total_seconds()/3600.0
        
            if volume:
                if duration_meds == 0.0:
                    rate = volume
                else:
                    rate = volume / duration_meds
                print(volume, rate, duration_meds)
            else:
                print("We're fucked")
        elif med_start and final_check:
            med_stop = med_start + pd.Timedelta(hours=duration)
            print(med_stop, med_start)
        
            if not duration:
                print("We're fucked")
        
        return {
            "volume": volume,
            "rate": rate,
            "duration": duration,
            "med_start": med_start,
            "med_stop": med_stop,
            "med_action": med_action,
            "med_action_dose": med_action_dose,
            "med_action_dose_unit": med_action_dose_unit,
            "weight": weight,
        }
    


#Create a dataframe with the same index as the supertable, starting from first med_action_time 
start_time = pd.to_datetime(meds['med_action_time'].iloc[0])
extended_index = pd.date_range(
   start=start_time,
   end=supertable_index.iloc[-1], 
   freq='H'
)

medsdf = pd.DataFrame(index=extended_index)
medsdict = {}
ongoing_infusion = {}








