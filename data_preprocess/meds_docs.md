Emory:

1. matched infusion_meds flatfile with CJSEPSIS_OUT and ORDERED_MEDS as in merge_fluids_files.py 
    1. OUT contains some weird pull of only a specfic set of fluids but has "order clinical desc" which can sometimes be useful to see what rate/volume of fluids were ORDERED. You have to parse this clinical desc. Right now I am doing this using a piece of logic that some unknown person had come up with that was handed to me as a "solution to emory fluids". A slightly improved version of the code is in process_fluids.py. The result of running this code (fluid parameters) is already saved as columns in the CJSEPSIS_OUT3.dsv file that exists on the RCC and DCC, so no one really has to run it again. I have tested enough to know that it works okay enough (i.e., parses the clinical description accurately) for 80% of the cases but fails in many important cases. It uses a rule based NLP approach - maybe can be improved. But the bigger problem is not the algorithm is rudimentary, it is that this file a) contains only specific fluids, b) does not even match the infusion_meds flattfile on most instances that this fluid was given -- the infusion meds file has many recordings of these fluids that are not recorded in this file. Idk if that's because this file was pulled for just a specific cohort or bed location, and c)Even for csns for whcih this file has recordings, the information derived from it is sometimes different from what was actually infused (given that my way of figuring out what was actually infused is correct). I have not yet calculating how many cases there is a descrepency for.
    2. ORDERED_MEDS is more useful. It contains the exact rate, duration, and volume that meds were ordered for without having to parse a clinical desc. These parameters usually match what was infused so it is a good way to verify that the infusion_meds information-derivation algorithm is correct. Again, these orders exists for just a subset. I have no idea about the origin of this file either so can't tell if it was only for a specific cohort.
2. The "formulary name" of the medication is important in our pipeline to extract reliable information (step 3 onwards). Some rows have formulary name as "Not Recorded" even though they record a valid med action. But they can have thier formulary name recorded in subsequent (or earlier rows). I've written this gnarly function called impute_by_closest_location() in medication_processor.py, which is what it says it is, that is, imputed formulary name by closest row (according to med_action_time) that shares the same order_med_id and med_name. Maybe this can be optimized because it takes a long while to run (4 hours for 10 ish million rows hehe) but I can't spend time and brain cells on that right now.  UPDATE: impute_by_closest_location_vectorized() is faster (5 mins for 10 ish million rows).
3. I parse the "formulary_name" of the merged file to do the following:
    1. figure out if the formulary name is for ANY kind of fluid (classify_fluids.py --> is_fluids column in em_infusion_meds_classification_final.csv) [Reason: there are some meds that are oral/synringes - didnt want them to mess with my "total amount of fluids administered" logic, but this is_fluids column ended up not being that useful in my final code]
    2. flag anesthaesia meds --> is_anes column in em_infusion_meds_classification_final.csv [Reason: anesthaesia meds dont have trustworthy volume or rate recordings]
    3. flag is_infusion --> includes all meds and fluids given as infusions (classify_fluids). Result in em_infusion_meds_classification_final.csv
    4. Classify meds into types with the help of Dr. Victor Moas with the final classification in em_infusion_meds_classification_final.csv [For eventually classifying types of meds given]
    5. I also get the "volume" (and unit), "amount" (and unit) from the formulary name and convert it to standard units. This is helpful when calculating how much fluid was administered and at what rate, for med orders that don't have matching entries in CJSEPSIS_OUT and ORDERED_MEDS (talked about in point 1 above) 
    6. Finally, some of these formulary names are not at all informative and dont have volumes or concentrations so it is very difficult to either figure out how much fluid was given or at what rate (for example, DOBUTamine/d5w tells me nothing) so Dr. Andre Holder created a default mapping in consultation with an emory pharmacist (for example, DOBUTamine/d5w is by default now assumed to be given at 500 mg / 250 mL unless there is an explicit order description matching the order id). These were used to get volumes and/or concentrations.
4. Next, I do some cohort filtering. Any encounter that has a bed_label in
    NURSERY INTENSIVE        
REHAB 4-BED              
NURSERY INTERMEDIATE     
NURSERY LEVEL 2          
OBSERVATION              
NURSERY                  
was filtered out because they had unreliable data and very few csns 

5. Fluids Algorithm:
    1. Filter out non_infusion_meds (don't contribute much to fluid volume)
    2. Filter out all rows that say premix (medication_processor.py : process_premix()) -- these are diluents I think but these rows don't give us any fluids information. They are usually coupled with the actual meds rows which give you the volume of premix. 
    3. Filter out all rows that have a nan or "NOT RECORDED" formulary name and no other match on the associated order_med_id. Sometimes diluents are "not recorded" because the associated med has the info. My algorithm should handle those cases
    4. 

