Jack Regan
02-23-2025

Program Documentation Notes:

- system requires dsv files in intiial processing pipeline. Potential area to use reflection in order to accomodate other files
    - QUESTION: What other file types are common in EMR data? Is this something we know?
    - TODO: accomodate all potential file signatures (different types of signatrues (ie, DSV, HL7, FEC))

- system requires the following data: infusion meds, labs, vitals, ventilator information, demograhics, GCS, encoutners, cultures, bed locations, procedures, and diagnosis. Each of these variables should be stored in their own dsv and labeled as such. These dsv's are seperated by year
    - TODO: probe user for number/iteration of years to pull from

- First Step: create a pickle by saving yearly variable values to dictionaries for each year. The items of the dictionary are dataframes containing relevant information from the previously mentioned dsv files. Most creations of dataframes starts with reading the csv file, dropping unused data, converting numeric data to number variables, merging predefined columns. The df is then dumped to the pickle.
    - TODO: explore inefficiency in sepyIMPORT hard coded values by variable type and how this can be standardized.

- Second Step: pull the pickled data and write as dictionaries. Dictionaries are saved to an output file that is deliminated by year
    - QUESTION: what additional information is added in the make_dicts step (Lines 194-206 of make_dicts)

    - TODO: add dialysis history to pickle step and avoid additional python script (highly inefficient to re-read the supertable)


    - TODO: add proper documentation to each step and clean up formating


April 4:
- a lot of variablility in file format (csv, xlsx) - better to conform to csv?
- store cohort selector values in configuration file (351-356)

- What are specifics read in through bash (324 and 330)




March 21:
- import_demographics - make_pickle expects "dob" within CJSEPSIS_DEMOGRAPHICS_2014.dsv but doesn't exist. Should this be a different age identifier?
    - same applies to import_infusion_meds, "vasopressor", "anti-infective"
    - Stopped progression, expectred more errors will continue to occur


- Once config are done, make a sincle script that works for several datasets
- combine make dicts and make pickle
    - How is yearly pickle file and we are making encounters from that
    - consider more efficient methods of data storage










Old Notes:

- em_make_dicts.py Line 165-169 - why are some of the ICD10 dicts commented out?
        - Don't get rid of ICD 9, check if datast uses ICD 9 vs ICD 10
        - Most emr data will have ICD version in a column, column will typically be called "ICD" - check for this value and make a distinction between the two

- em_make_dicts.py Line 224 - why is columns "Unamed: 0" dropped, is this a common occurance or specific to the Emory dataset?

- em_make_dicts.py Line 229 - why ese read_excel function, will this always be an excel file? file is named csv?

- em_make_dicts.py Lines 195-220, 229-250 - need definitions of, specifically num_processes (what is a process)

- em_make_dicts.py Line 349 - Keep IC10 directory path commented out?

- gr_make_pickle.py Line 51-709 - year based data paths have weird path names, should this be made uniform with emory data points?
    - check DCC for proper file naming conventions. should be fairly consistent with emory dataset

- gr_make_pickle.py line 780 - read_pickle is called seemingly out of nowhere
    - replicate emory file and import labs (previosuly done for speed)

- gr_make_dicts.py Line 25 - why is DIALYSIS_INFO_CSN_FNAME set to none?

- gr_make_dicts.py Line 237-247 - add line from em pipeline to make consistent

- add_dialysis_history_script.py Line 62 - what is new new_sup_path, is this output path from make_dicts.py?
_______

- test directory - what are test files testing?

_______
- sepyIMPORT.py Line 182 - what are arguments and return values?
    - there certain columns in the emr data that are data columns, s are a list of "data times" in a particular column.


Suggestions:
em_make_pickle.py
- create a basline config file for each of column values, read into make_pickle based on import function type