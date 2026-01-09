Emory:

1. matched infusion_meds flatfile with CJSEPSIS_OUT3 and ORDERED_MEDS as in merge_fluids_files.py 
    1. OUT contains some weird pull of only a specific set of fluids but has "order clinical desc" which can sometimes be useful to see what rate/volume of fluids were ORDERED. You have to parse this free text clinical desc. Right now I am doing this using a piece of logic that some unknown person had come up with that was handed to me as a "solution to emory fluids" lololol. A slightly improved version of the code is in process_fluids.py. The result of running this code (fluid parameters) is already saved as columns in the CJSEPSIS_OUT3.dsv file that exists on the RCC and DCC, so no one really has to run it again. I have tested enough to know that it works okay enough (i.e., parses the clinical description accurately) for 80% of the cases but fails in many important cases. It uses a rule based NLP approach - maybe can be improved. But the bigger problem is not that the algorithm is rudimentary, it is that this file a) contains only specific fluids, b) does not even match the infusion_meds flatfile on most instances that this fluid was given -- the infusion meds file has many recordings of these fluids that are not recorded in this file. Idk if that's because this file was pulled for just a specific cohort or bed location, and c)Even for csns for whcih this file has recordings, the information derived from it is sometimes different from what was actually infused (given that my way of figuring out what was actually infused is correct). I have not yet calculated how many cases there is a descrepency for.
    2. ORDERED_MEDS is more useful. It contains the exact rate, duration, and volume that meds were ordered for without having to parse a clinical desc. These parameters usually match what was infused so it is a good way to verify that the infusion_meds information-derivation algorithm is correct. Again, these orders exists for just a subset. I have no idea about the origin of this file either so can't tell if it was only for a specific cohort.
2. The "formulary name" of the medication is important in our pipeline to extract reliable information (step 3 onwards). Some rows have formulary name as "Not Recorded" even though they record a valid med action. But they can have their formulary name recorded in subsequent (or earlier rows). I've written this gnarly function called impute_by_closest_location() in medication_processor.py, which is what it says it is, that is, imputes formulary name by closest row (according to med_action_time) that shares the same order_med_id and med_name. Maybe this can be optimized because it takes a long while to run (4 hours for 10 ish million rows hehe) but I can't spend time and brain cells on that right now.  UPDATE: I remembered that I can & have already spent $20 on cursor so now we have impute_by_closest_location_vectorized() that is is faster (5 mins for 10 ish million rows).
3. I parse the "formulary_name" of the merged file to do the following:
    1. figure out if the formulary name is for ANY kind of fluid (classify_fluids.py --> is_fluids column in em_infusion_meds_classification_final.csv) [Reason: some meds are diluted in fluids, and the way to determine the concentration of the medicine is to find the rate at which the fluid was given and use that to find the concentration]
    2. flag anesthaesia meds --> is_anes column in em_infusion_meds_classification_final.csv [Reason: anesthaesia meds dont have trustworthy volume or rate recordings]
    3. flag is_infusion --> includes all meds and fluids given as infusions (classify_fluids). Result in em_infusion_meds_classification_final.csv
    4. Classify meds into types with the help of Dr. Victor Moas with the final classification in em_infusion_meds_classification_final.csv [For eventually classifying types of meds given]
    5. I also get the "volume" (and unit), "amount" (and unit) from the formulary name and convert it to standard units. This is helpful when calculating how much fluid was administered and at what rate, for med orders that don't have matching entries in CJSEPSIS_OUT and ORDERED_MEDS (talked about in point 1 above) 
    6. Finally, some of these formulary names are not at all informative and dont have volumes or concentrations so it is very difficult to either figure out how much fluid was given or at what rate (for example, DOBUTamine/d5w tells me nothing) so Dr. Andre Holder created a default mapping in consultation with an emory pharmacist (for example, DOBUTamine/d5w is by default now assumed to be given at 500 mg / 250 mL unless there is an explicit order description matching the order id). These are populated in the concentration_default column. Some meds instead have a default rate, which should be used unless there is an order-specified rate. 
    7. For formulary names that have a concentration_default (for formulary names that dont specify volume), I first parse the concentration_default:
        1. Extract amount, amount unit, and volume from concentration_default
        2. Convert the extracted amount to milligrams (volume remains unchanged - concentration ratio adjusts automatically)
    Then, I parse the formulary name to check if they have an amount. 
    If formulary has amount:
        1. I map formulary name amount to milligrams
        2. Calculate volume administered using: volume = amount_in_mg / concentration_in_mg_per_ml
    If formulary name does not have an amount:
        1. Use concentration_default volume as volume administered
    When I am finally processing rows, I first check if there is a parent order rate/volume for a med, then if there is volume//rate from the formulary name, then use the volume/rates derived in this step from the concentration_defaults. 
    8. Repeat step 7 but for rate_defaults (which is always in ml/hr)
4. Next, I do some cohort filtering. Any encounter that has a bed_label in
    NURSERY INTENSIVE        
REHAB 4-BED              
NURSERY INTERMEDIATE     
NURSERY LEVEL 2          
OBSERVATION              
NURSERY                  
SPECIAL_CARE (these were all non-icu csns and most seemed to be receiving chemo)
was filtered out because they had unreliable data and very few csns 
5. Then I drop the rows with "formulary name" not recorded and order_med_id not associated with any other med type.

6. Fluids Algorithm:
   For each unique csn:
    1. Initialize a all_meds_dict with keys order_med_id and a subdictionary with fields: formulary_name, med_name, med_class, med_subclass, med_start_time, med_stop_time. This is dictionary which will create indicator columns for when a particular medication was administered (w/o going into dose details).
    2. Initialize another fluid_medsdict with order_med_id as keys. This will track the rates, volumes etc. of the medications/fluids that contribute to volume.
    3. Sort the csn-sliced df by med_action_time. 
    4. Remove the premix meds (They dont give us any information)
    5. Get unique order_med_ids, process each sequentially
    6. For each unique order_med_id:
        1. Get unique med_names associated with that order_med_id
        If one unique_med_name:
            If not is_infusion:
                add to all_meds_dict. If order_med_id already exists, append the start and stop time of this instance to the "med_start_time" and "med_stop_time" fields (which should be lists)
            If infusion:
                run process_medication_timeline_new() defined in medication_processor.py. This documented further below. 
        If two unique_med_names:
            If any of the two is not is_infusion:
                Process the not is_infusion: 
                    add to all_meds_dict. If order_med_id already exists, append the start and stop time of this instance to the "med_start_time" and "med_stop_time" fields (which should be lists)
                process the other one: 
                    if the formulary name is "Not Recorded" : 
                        skip (it was used to dilute the med in the synringe or injection so it's a nominal volume)
                    else: 
                        run process_medication_timeline_new() defined in medication_processor.py. This documented further below.
            
            else:
                process as dilution pairs (method documented below)
        If more than two unique meds:
            NotImplementedError
        
        process_medication_timeline_new():
            LOGIC:
             1. **Words used in these docs**: 
                - **"initialized"**: Whether this medication has been successfully processed before (stored params exist to reuse)
                - **"original params"**: First successfully extracted rate/volume/duration for this medication, stored for reuse in later rows
                - **get_order_params_for_row()**: Attempts to extract rate/volume/duration by: (a) checking ORDERED_MEDS, (b) parsing ORDER_CLINICAL_DESC text, (c) reconciling conflicts between sources, (d) using formulary volume/amounts, (e) applying fallbacks. Returns extracted params + "final_check" flag indicating confidence.
                - **"final_check"**: Boolean indicating extracted parameters passed validation (internally consistent, from reliable sources)
                - **actual_duration**: Real duration calculated from subsequent "Infuse" action timestamps (more accurate than ordered duration)
             
             2. **Processing by med_action type** (rows sorted by time):
             
             **Begin Bag** (starts new infusion bag):
             - **First Begin Bag** (medication not yet initialized):
               1. Extract params via get_order_params_for_row()
               2. Find actual duration by locating last "Infuse" action after this Begin Bag
               3. If extraction succeeded (final_check=True):
                  - Store params and save as "original params" for future rows
                  - May recalculate rate if: volume is known AND (rate is missing OR rate was based on 1hr assumption for small fluids)
               4. If extraction failed (final_check=False):
                  - Try defaults: volume_from_concentration or rate_default_numeric
                  - If defaults found: initialize and save as "original params"
                  - If no defaults: skip this medication
             
             - **Subsequent Begin Bags** (medication already initialized):
               1. Extract params via get_order_params_for_row()
               2. Find actual duration from Infuse actions
               3. If extraction succeeded: use newly extracted params
               4. If extraction failed: reuse "original params" from first bag
               5. Always use actual duration (if found) over ordered duration (derived from OUT3 or ORDERED_MEDS)
               6. Append this infusion period to medication's timeline
             
             **Rate Change** (changes infusion rate mid-administration):
             1. If medication not initialized: Treat as implicit Begin Bag
             2. Extract params via get_order_params_for_row()
                - If succeeded: use new params
                - If all params = 0: medication stopped
                - If failed: reuse "original params"
             3. If new rate equals previous rate: ignore (no actual change)
             4. **If rate change during active infusion**:
                - Calculate volume already delivered and remaining volume
                - End previous infusion period at rate change time
                - Start new period with remaining volume at new rate
                - Find actual duration for new period from Infuse actions
             5. **If rate change after infusion ended** (orphaned): Treat as implicit Begin Bag
             
             **Infuse** (ongoing administration documentation):
             - **Purpose**: Used to calculate actual duration for Begin Bag/Rate Change actions. Most Infuse rows are consumed during duration calculation.
             - **If Infuse is first action** (medication not initialized):
               1. Check if Begin Bag exists within 1 hour ahead
               2. If yes: ignore this Infuse (it will be consumed by that Begin Bag)
               3. If no: treat as implicit Begin Bag
             - **If Infuse not consumed** (orphaned):
               1. Check if within any active infusion period
               2. If yes: mark as processed (normal infuse during active period)
               3. If no: treat as implicit Begin Bag
             
             **Not Recorded** (medication stop, has explicit med_start/med_stop times):
             1. Calculate duration from row's med_start and med_stop timestamps
             2. Extract params via get_order_params_for_row()
             3. **If extraction succeeded** (final_check=True):
                - Use extracted rate/volume
                - Calculate volume from rate × duration if missing
                - If medication not initialized: save as "original params"
                - Otherwise: append period to timeline
             4. **If extraction failed but medication initialized**:
                - Use "original params" rate
                - Calculate volume = rate × duration
                - Append period to timeline
             5. **If extraction failed and medication NOT initialized**:
                - Try defaults: volume_from_concentration or rate_default_numeric
                - Calculate missing param (rate from volume/duration or vice versa)
                - If defaults found: initialize with defaults as "original params"
                - If no defaults: skip
             
             **Bolus** (rapid/push administration):
             1. Extract params via get_order_params_for_row()
             2. **If extraction succeeded** (final_check=True):
                - Use extracted params
                - Calculate volume from rate × duration if missing
                - Med_stop = start + duration
             3. **If extraction failed**:
                - Assume duration = 1 hour
                - Try to find volume/rate from: (a) extracted volume even if final_check=False, (b) "original params" if initialized, (c) defaults (volume_from_concentration or rate_default_numeric)
                - Calculate missing param: rate from volume/1hr OR volume from rate × 1hr
                - Med_stop = start + 1hr
             4. If medication not initialized: save as "original params". Otherwise: append period.
             
             **Other actions** (unexpected action types):
             - If medication not initialized:
               1. Extract params via get_order_params_for_row()
               2. If succeeded: treat as implicit Begin Bag
               3. Otherwise: ignore
             - If medication initialized: ignore
             
             **Implicit Begin Bag logic** (used for orphaned Rate Changes, Infuses, other actions):
             1. Extract params via get_order_params_for_row()
             2. Find actual duration from Infuse actions
             3. Determine params to use:
                - If extraction succeeded: use extracted params
                - If all params = 0: skip (medication stopped before starting)
                - If failed but medication initialized: use "original params"
                - If failed and not initialized: skip
             4. Prefer actual duration over ordered duration
             5. May recalculate rate if volume exists and rate was 1hr assumption
             6. If medication not initialized: initialize and save as "original params". Otherwise: append period.
             
             3. **get_order_params_for_row() details**:
                a. extract_parent_order_params(): Get volume/rate/duration from ORDERED_MEDS structured fields. Flag "suspicious" if wrong units.
                b. extract_clinical_desc_params(): Parse ORDER_CLINICAL_DESC text with regex. Flag "sus" if conflicts/inconsistencies.
                c. reconcile_parameters(): Choose between parent vs clinical based on confidence, completeness, consistency. Returns final_check=True if confident.
                d. extract_infusion_params(): Get volume_inf, amount_inf, med_action_dose from formulary columns. Convert dose rates to mL/hr.
                e. Fallback hierarchy (if reconciliation failed or rate missing): Try infusion_params with unit conversion, then volume_inf with duration, then 1hr assumption for small fluids.
                f. Duration validation: If med_stop exists, override with actual duration.


 

7. Add diuretics from non_infusion meds 



## check

1. csn: 000a225932978247bb07bb4284713d8cedb74f8635f121be2b7b7eebecd24aa7
-- what happens when inf duration is 0?
-- dilution pairs error 

2. 0013d826efd44adcdc4ca4b021c335169d7f34f196d7d63d275784bed7c0c17c
-- all cases
--order id: 9630480923 : should the previous params be carried forward? 

3. check:
001bb660d966139adfd5afc734339902631e73ddaf4f552271e12147aa45ac08  - heparin

## todo

1. if med in dilution pair does not have a rate then use volume of the fluid and actual duration of infusion (like insulin)
2. dilution pairs: 
    check if fluid params are all correct otherwise use med params and only volume from the fluid
3. You need to handle non is_infusion meds also as dilution pairs because some pairs have rate!!
csn:
0013d826efd44adcdc4ca4b021c335169d7f34f196d7d63d275784bed7c0c17c
4. Okay if the med action is "Not Recorded" then the non _infusion med can be considered as not having rte information (need not be handled like a dilution pair) 
but volume needs to be added? check : 02412076954e45eef9a01604d08752ca2b4634bd29c23e83a6a4e2b3d0ad5d63

5. If formulary_name is not recorded - ignore the fluid in the pair

6. WHAT HAPPENS WHEN RATE IS NONE BUT VOLUME IS PRESENT AND IS INFUSION - FIXED 

7. Need to fix: med rate but calcualted using med volume - should use fluid volume for diuent pair

8. If infusion period code fails, at least add the med to the all_meds_dict


Dilution pairs:

1. Find common times between fluid and meds. 
2. Need to process non-common also - sometimes they have volume
(csn: 000cb42699f120ddf5dedfcaf4b6b5575055e12f43ae99b3af999b17ef99469b)
3. Calculate actual infused duration using med because that has the real rate 
0013d826efd44adcdc4ca4b021c335169d7f34f196d7d63d275784bed7c0c17c
(but get volume from fluid)
4. Central hyparel csn: 026030c3ce36817d1ae017799be0cf2cf374b64e1bdde50a764aeb7d83dac84b
hyperal peripheral : 026030c3ce36817d1ae017799be0cf2cf374b64e1bdde50a764aeb7d83dac84b
