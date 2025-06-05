# -*- coding: utf-8 -*-
"""

Elite Data Hacks
Author: Christopher S. Josef, MD
Email: csjosef@krvmail.com

Kamaleswaran Labs
Author: Jack F. Regan
Edited: 2025-03-06
Version: 0.2

Changes:
  - improved documentation
  - implemented yaml configuration file
"""

import time
import yaml
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
###########################################################################
################################ Load YAML ################################
###########################################################################
def load_yaml(filename):
    """
    Load and parse a YAML file.
    Args:
        filename (str): The path to the YAML file to be loaded.
    Returns:
        dict: The contents of the YAML file as a dictionary.
    """
    with open(filename, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
yaml_data = load_yaml("/cwork/jfr29/Sepy/configurations/import_config.yaml")
###########################################################################
############################### IMPORT Class ##############################
###########################################################################
class sepyIMPORT:
    """
    A class for importing and processing clinical data from CSV files, specifically designed
    to handle electronic medical records (EMR) datasets with various preprocessing steps.

    Args:
        na_values (list): A list of string values representing missing data in the dataset.
        vital_col_names (list): Column names related to vital signs.
        vasopressor_units (list): Column names representing units for vasopressor medications.
        numeric_lab_col_names (list): Column names for numerical lab results.
        string_lab_col_names (list): Column names for categorical lab results.
    """
    def __init__(self, file_dictionary, delim, config_data):
        # delimiter used in raw files can changea across data sets; the delim argument is specified by user
        self.delim = delim
        # dictionary has file locations for flat files
        self.file_dictionary = file_dictionary

        # creates df with all medication groupings
        self.df_grouping_all_meds = pd.read_csv(file_dictionary[config_data["dictionary_paths"]["grouping_types"][0]])
        # creates df with all lab groupings
        self.df_grouping_labs = pd.read_csv(file_dictionary[config_data["dictionary_paths"]["grouping_types"][1]])
        # creates df with all bed location labels
        self.df_bed_labels = pd.read_csv(file_dictionary[config_data["dictionary_paths"]["grouping_types"][2]])
    
        # for use when importing CSVs
        self.na_values = yaml_data["na_values"]
        # vital data type dictionary
        self.vital_col_names = yaml_data["vital_col_names"]
        # Vasopressor units
        self.vasopressor_units = yaml_data["vasopressor_units"]
        # List of all lab names (some years might not have all listed labs)
        self.numeric_lab_col_names = yaml_data["numeric_lab_col_names"]
        # List of all lab names (some years might not have all listed labs)
        self.string_lab_col_names = yaml_data["string_lab_col_names"]
        self.all_lab_col_names = self.numeric_lab_col_names + self.string_lab_col_names
        logging.info("sepyIMPORT initialized")

###########################################################################
############################## Data Cleaning ##############################
###########################################################################
    def make_numeric(self, df, cols):
        """  
        Cleans and converts specified columns in a DataFrame to numeric format.  
        Args:  
            df (pandas.DataFrame): The DataFrame containing the columns to be processed.  
            cols (list): A list of column names to clean and convert to numeric values.  
        Returns:  
            pandas.DataFrame: The modified DataFrame with specified columns converted to numeric types.  
        """
        # Remove all the non-numeric characters from numerical cols
        df[cols] = df[cols].replace(r"\>|\<|\%|\/|\s", "", regex = True)
        # Converts specific cols to numeric
        df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
        return df
###########################################################################
### Custom Date Parser to Handle Date Errors (i.e. coerce foolishness) ####
###########################################################################
    def d_parser(self, s):
        """  
        Parses a given string or array-like object into a datetime format.  
        Args:  
            s (str, list, or pandas.Series): The input data to be converted to datetime.  
        Returns:  
            pandas.Series or pandas.DateTimeIndex: The parsed datetime object(s).  
        """
        return pd.to_datetime(s, errors = "coerce")
###########################################################################
##################### Create encounter df for all CSNs ####################
###########################################################################
    def import_encounters(self, drop_cols, index_col, date_cols):
        """
        Imports the encounters dataset from a CSV file, parses date columns, 
        drops specified columns, and stores the cleaned DataFrame.

        Args:
            drop_cols (list o str): Columns to drop from the DataFrame.
            index_col (str): Column to use as the DataFrame index.
            date_cols (list of str): Columns to parse as datetime objects.
        """
        logging.info("Begin Encounters Import")
        path_encounters_file = self.file_dictionary["ENCOUNTER"]

        # import file and set date/time cols
        df_encounters = pd.read_csv(
            path_encounters_file,
            header=0,
            index_col=index_col,
            parse_dates=date_cols,
            na_values=self.na_values,
            sep=self.delim,
        )


        # drop unecessary cols
        df_encounters = df_encounters.drop(columns=drop_cols)
        self.df_encounters = df_encounters
        logging.info("Encounters success")
###########################################################################
#################### Create demographic df for all CSNs ###################
###########################################################################
    def import_demographics(self, drop_cols, index_col, date_cols):
        """
        Imports the demographics dataset from a CSV file, parses date columns, 
        drops specified columns, and stores the cleaned DataFrame.

        Args:
            drop_cols (list of str): Columns to drop from the DataFrame.
            index_col (str): Column to use as the DataFrame index.
            date_cols (list of str): Columns to parse as datetime objects.
        """
        logging.info("Begin Demographics Import")
        path_demographics_file = self.file_dictionary["DEMOGRAPHICS"]
        # import files and set date/time cols
        df_demographics = pd.read_csv(
            path_demographics_file,
            header=0,
            index_col=index_col,
            parse_dates=date_cols,
            na_values=self.na_values,
            sep=self.delim,
            low_memory=False,
        )
        # drop unecessary cols
        df_demographics = df_demographics.drop(columns=drop_cols)

        self.df_demographics = df_demographics
        logging.info("Demographics success")
###########################################################################
########### Create df of filtered medications for all CSNs ################
###########################################################################
    def import_infusion_meds(
        self,
        drop_cols,
        numeric_cols,
        anti_infective_group_name,
        vasopressor_group_name,
        index_col,
        date_cols,
    ):
        """
        Imports the infusion medication dataset from a CSV file, processes the data by 
        converting specified columns to numeric, removes unnecessary columns, checks for 
        duplicates in medication IDs, and separates the data into anti-infective and 
        vasopressor medication groups.

        Args:
            drop_cols (list of str): Columns to drop from the DataFrame.
            numeric_cols (list of str): Columns to convert to numeric values.
            anti_infective_group_name (str): The name of the anti-infective medication group.
            vasopressor_group_name (str): The name of the vasopressor medication group.
            index_col (str): Column to use as the DataFrame index.
            date_cols (list of str): Columns to parse as datetime objects.
        """
        path_infusion_med_file = self.file_dictionary["INFUSIONMEDS"]

        # imports the infusion med file and sets date/time cols
        read_infusion_csv_time = time.time()
        logging.info("Starting csv read for infusion meds.")

        df_infusion_meds = pd.read_csv(
            path_infusion_med_file,
            header=0,
            parse_dates=date_cols,
            index_col=index_col,
            sep=self.delim,
            na_values=self.na_values,
            low_memory=False,
            memory_map=True,
        )
        logging.info(
            f"It took {time.time()-read_infusion_csv_time} (s) to read the infusion csv."
        )

        df_infusion_meds = self.make_numeric(df_infusion_meds, numeric_cols)
        logging.info("Infusion columns are now numeric")

        # drop unecessary columns
        self.df_infusion_meds = df_infusion_meds.drop(columns=drop_cols)

        # check if there are duplicate med id's & print warining if there is error
        rows_dropped = (
            self.df_grouping_all_meds.shape[0]
            - self.df_grouping_all_meds.drop_duplicates("medication_id").shape[0]
        )
        if rows_dropped > 0:
            # drop the duplicate rows in grouping all meds
            self.df_grouping_all_meds = self.df_grouping_all_meds.drop_duplicates(
                subset="medication_id"
            )
            logging.info(
                f"You have {rows_dropped} duplicates of medication ID that were dropped!"
            )
        else:
            logging.info("Congrats, You have NO duplicates of medication ID!")

        # get med_ids for anti_infective
        df_anti_infective_med_groups = self.df_grouping_all_meds[
        self.df_grouping_all_meds["med_class"] == anti_infective_group_name
        ][["super_table_col_name", "medication_id"]]

        # makes df with anti-infective meds
        self.df_anti_infective_meds = (
            df_infusion_meds.reset_index()
            .merge(df_anti_infective_med_groups, how="inner", on="medication_id")
            .set_index("csn")
        )

        logging.info("Anti-infective success")

        # get med_ids for vassopressors
        df_vasopressor_med_groups = self.df_grouping_all_meds[
        self.df_grouping_all_meds["med_class"] == vasopressor_group_name
        ][["super_table_col_name", "medication_id"]]

        # makes df with vasopressor ; adds a numerically increasing index along with csn and supertable name
        # only keeps the dose and dose unit cols
        df_vasopressor_meds = (
            df_infusion_meds.reset_index()
            .merge(df_vasopressor_med_groups, how="inner", on="medication_id")
            .reset_index()
            .set_index(["csn", "med_order_time", "super_table_col_name"], append=True)[
                ["med_action_dose", "med_action_dose_unit"]
            ]
        )

        # unstack the units
        units = df_vasopressor_meds["med_action_dose_unit"].unstack(level=3)
        # unstack the dose
        dose = df_vasopressor_meds["med_action_dose"].unstack(level=3)
        # merge the dose and units together
        df_vasopressor_meds = dose.merge(
            units, left_index=True, right_index=True, suffixes=["", "_dose_unit"]
        )

        # unstack makes a multi-index for columns; this line removes "lab result" level
        # df_vasopressor_meds.columns = df_vasopressor_meds.columns.droplevel()

        # this removes the "name" for all the columns i.e. super_table_col_names
        df_vasopressor_meds.columns.name = None

        # removes numerical index for vasopressor meds
        df_vasopressor_meds = df_vasopressor_meds.droplevel(0)

        # drops med_order_time as index
        df_vasopressor_meds = df_vasopressor_meds.reset_index(1)

        # cols that don't units will break "groupby" related actions later, so need to remove nan
        df_vasopressor_meds[self.vasopressor_units] = df_vasopressor_meds[
            self.vasopressor_units
        ].replace({np.nan: ""})

        self.df_vasopressor_meds = df_vasopressor_meds
        logging.info("Vasopressor success")
###########################################################################
####################### Create lab df for all CSNs ########################
###########################################################################
    def import_labs(self, drop_cols, group_cols, date_cols, index_col, numeric_cols):
        ### The labs import needs a special function to tiddy the columns
        def tidy_index(df):
            # turns super_table_name into a col
            df = df.unstack(level=1)
            # unstack makes a multi-index for columns; this line removes "lab result" level
            df.columns = df.columns.droplevel()
            # this removes the "name" for all the columns i.e. super_table_col_names
            df.columns.name = None
            # removes numerical index for labs
            df = df.droplevel(0)
            return df

        # start timer for the lab import function
        start_import_time = time.time()
        logging.info("Begin Lab Import")
        # Lab groups file has three important cols:
        # 1) component_id - the is the id number for the lab type
        # 2 )super_table_col_name- is the group name for similar labs
        # 3) physionet - indicates if used for physionet competition

        lab_groups = self.df_grouping_labs[group_cols]
        
        path_lab_file = self.file_dictionary["LABS"]

        # import the lab flat file and set date/time
        df_labs = pd.read_csv(
            path_lab_file,
            header=0,
            parse_dates=date_cols,
            date_parser=self.d_parser,
            na_values=self.na_values,
            sep=self.delim,
            low_memory=False,
            memory_map=True,
        )

        # drop unecessary cols
        df_labs = df_labs.drop(columns=drop_cols)

        ### Select Relavent Lab Groups ###
        lab_groups = lab_groups[["super_table_col_name", "component_id"]][
            lab_groups["import"] == "Yes"
        ]

        ### Join Groups and Lab File ###
        df_labs_filtered = df_labs.merge(lab_groups, how="inner", on="component_id")

        # if there is no collection time, use result time - 1hr
        df_labs_filtered["collection_time"] = df_labs_filtered[
            "collection_time"
        ].fillna(df_labs_filtered["lab_result_time"] - pd.Timedelta(hours=1))

        # set index (necessary for unstacking)
        df_labs_filtered.set_index(
            [
                "super_table_col_name",
                "csn",
                "component_id",
                "result_status",
                "lab_result_time",
                "collection_time",
                "pat_id",
                "proc_cat_id",
                "proc_cat_name",
                "proc_code",
                "proc_desc",
                "component",
                "loinc_code",
            ],
            append=True,
            inplace=True,
        )
        
        # Select Labs that have string value
        # isolate string lab value rows
        df_labs_filtered_string = df_labs_filtered.loc[
            df_labs_filtered.index.get_level_values("super_table_col_name").isin(
                self.string_lab_col_names
            )
        ]
        
        # Select and Treat Labs that have Numeric value
        # isolate numeric lab value rows
        df_labs_filtered_numeric = df_labs_filtered.loc[
            df_labs_filtered.index.get_level_values("super_table_col_name").isin(
                self.numeric_lab_col_names
            )
        ]

        # remove punctuation from numeric
        df_labs_filtered_numeric = df_labs_filtered_numeric.replace(
            r"\>|\<|\%|\/|\s", "", regex=True
        )

        # convert labs to numeric
        df_labs_filtered_numeric["lab_result"] = pd.to_numeric(
            df_labs_filtered_numeric["lab_result"], errors="coerce"
        )

        # Tiddy up index using previously defined fcn
        df_labs_numeric = tidy_index(df_labs_filtered_numeric)
        df_labs_string = tidy_index(df_labs_filtered_string)

        logging.info(
            f"It took {time.time() - start_import_time}(s) to import and process labs."
        )

        df_labs_numeric = df_labs_numeric
        df_labs_string = df_labs_string
        # Concat the string and numeric dfs
        df_labs_all = pd.concat([df_labs_numeric, df_labs_string], axis=0)
        
        # Remove duplicate columns
        df_labs_all = df_labs_all.loc[:, ~df_labs_all.columns.duplicated()]
    
        # if there are missing cols (i.e. no COVID in 2014) then it ensures the col name is added
        self.df_labs = df_labs_all.reindex(
            df_labs_all.columns.union(self.all_lab_col_names), axis=1
        )
        logging.info("Labs success")
###########################################################################
###################### Create vitals df for all CSNs ######################
###########################################################################
    def import_vitals(self, drop_cols, index_col, date_cols, merge_cols):
        """
        Imports the lab dataset from a CSV file, processes both numeric and string lab values, 
        applies necessary transformations, and stores the cleaned DataFrame.

        Args:
            drop_cols (list of str): Columns to drop from the DataFrame.
            group_cols (list of str): Columns to group labs by for processing.
            date_cols (list of str): Columns to parse as datetime objects.
            index_col (str): Column to use as the DataFrame index.
            numeric_cols (list of str): Columns that contain numeric lab results.
        """
        logging.info("Begin Vitals Import")

        path_vitals_file = self.file_dictionary["VITALS"]

        # import vitals flat file and set date/time cols
        df_vitals = pd.read_csv(
            path_vitals_file,
            header=0,
            dtype=object,
            parse_dates=date_cols,
            index_col=index_col,
            na_values=self.na_values,
            sep=self.delim,
            low_memory=False,
        )
        # drop unecssary cols
        df_vitals = df_vitals.drop(columns=drop_cols)

        # If there are columns to merge then do this:
        if merge_cols is not None:
            for merge_set in merge_cols:
                df_vitals[merge_set[2]] = df_vitals[merge_set[0]].fillna(
                    df_vitals[merge_set[1]]
                )
                df_vitals = df_vitals.drop(columns=[merge_set[0], merge_set[1]])

        # drop punctuation and make numeric
        start_to_numeric_conversion_time = time.time()  # start timer
        df_vitals = self.make_numeric(df_vitals, sepyIMPORT.vital_col_names)
        logging.info(
            f"It took {time.time()-start_to_numeric_conversion_time} to convert vitals results to numeric."
        )

        self.df_vitals = df_vitals
        logging.info("Vitals success")
###########################################################################
####################### Create vent df for all CSNs #######################
###########################################################################
    def import_vent(self, drop_cols, numeric_cols, index_col, date_cols):
        """
        Imports the ventilation dataset from a CSV file, processes numeric columns, 
        drops unnecessary columns, and stores the cleaned DataFrame.

        Args:
            drop_cols (list of str): Columns to drop from the DataFrame.
            numeric_cols (list of str): Columns to convert to numeric values.
            index_col (str or int): Column to use as the DataFrame index.
            date_cols (list of str): Columns to parse as datetime objects.
        """
        logging.info("Begin Vent Import")
        path_vent_file = self.file_dictionary["VENT"]

        # import vent flat files and set date/time
        df_vent = pd.read_csv(
            path_vent_file,
            header=0,
            parse_dates=date_cols,
            index_col=index_col,
            na_values=self.na_values,
            sep=self.delim,
            low_memory=False,
        )

        # drop unecessary cols
        df_vent = df_vent.drop(columns=drop_cols)

        df_vent = self.make_numeric(df_vent, numeric_cols)

        self.df_vent = df_vent
        logging.info("Vent success")
###########################################################################
####################### Create GCS df for all CSNs ########################
###########################################################################
    def import_gcs(self, drop_cols, index_col, numeric_col, date_cols):
        """
        Imports the GCS dataset from a CSV file, processes the numeric columns, 
        aggregates GCS scores by timestamp, and stores the cleaned DataFrame.

        Args:
            drop_cols (list of str): Columns to drop from the DataFrame.
            index_col (str): Column to use as the DataFrame index.
            numeric_col (list of str): Columns to convert to numeric values.
            date_cols (list of str): Columns to parse as datetime objects.
        """
        logging.info("Begin GCS Import")

        path_gcs_file = self.file_dictionary["GCS"]

        # import gcs flat file and set date/time cols
        df_gcs = pd.read_csv(
            path_gcs_file,
            header=0,
            parse_dates=date_cols,
            index_col=index_col,
            na_values=self.na_values,
            sep=self.delim,
            low_memory=False,
        )

        # drop unecessary cols
        df_gcs = df_gcs.drop(columns=drop_cols)

        # ensure all score coloumns are numeric
        self.make_numeric(df_gcs, numeric_col)

        # merges all gcs values into a single timestamp/row
        df_gcs = df_gcs.groupby(["csn", "recorded_time"]).aggregate(
            {
                "gcs_eye_score": ["mean"],
                "gcs_verbal_score": ["mean"],
                "gcs_motor_score": ["mean"],
                "gcs_total_score": ["mean"],
            }
        )
        # drops the column index "mean" which came from agg fcn
        # also moves 'recorded time' out of index into a column
        df_gcs = df_gcs.droplevel(1, axis=1).reset_index(level="recorded_time")

        self.df_gcs = df_gcs
        logging.info("GCS success")
###########################################################################
#################### Create cultures df for all CSNs ######################
###########################################################################
    def import_cultures(self, drop_cols, index_col, date_cols):
        """
        Imports the cultures dataset from a CSV file, processes date columns, 
        drops unnecessary columns, and stores the cleaned DataFrame.

        Args:
            drop_cols (list of str): Columns to drop from the DataFrame.
            index_col (str or int): Column to use as the DataFrame index.
            date_cols (list of str): Columns to parse as datetime objects.
        """
        logging.info("Begin Cultures Import")

        path_cultures_file = self.file_dictionary["CULTURES"]

        # imports cultures and sets date/time cols
        df_cultures = pd.read_csv(
            path_cultures_file,
            header=0,
            parse_dates=date_cols,
            index_col=index_col,
            na_values=self.na_values,
            sep=self.delim,
            low_memory=False,
        )

        # drops unecessary cols
        df_cultures = df_cultures.drop(columns=drop_cols)

        self.df_cultures = df_cultures
        logging.info("Cultures success")
###########################################################################
################## Create bed_location df for all CSNs ####################
###########################################################################
    def import_bed_locations(self, drop_cols, index_col, date_cols):
        """
        Imports the bed locations dataset from a CSV file, processes date columns, 
        adds identifiers for various bed types (ICU, IMC, ED, procedure), 
        filters out duplicate rows, drops unnecessary columns, and stores the cleaned DataFrame.

        Args:
            drop_cols (list of str): Columns to drop from the DataFrame.
            index_col (str): Column to use as the DataFrame index.
            date_cols (list of str): Columns to parse as datetime objects.
        """
        logging.info("Begin Bed Import")

        bed_labels = self.df_bed_labels
        path_bed_locations_file = self.file_dictionary["BEDLOCATION"]

        # import bed flat files and set date/time cols
        df_beds = pd.read_csv(
            path_bed_locations_file,
            header=0,
            parse_dates=date_cols,
            index_col=index_col,
            na_values=self.na_values,
            sep=self.delim,
            low_memory=False,
        )

        # drop anytimes where bed_location_start = bed_location_end
        df_beds = df_beds[df_beds["bed_location_start"] != df_beds["bed_location_end"]]

        # Identifier column for ICU bed
        icu_units = bed_labels[bed_labels["icu"] == 1].bed_unit.tolist()
        df_beds["icu"] = np.where(df_beds["bed_unit"].isin(icu_units), 1, 0)

        # Identifier column for IMC
        imc_units = bed_labels[bed_labels["imc"] == 1].bed_unit.tolist()
        df_beds["imc"] = np.where(df_beds["bed_unit"].isin(imc_units), 1, 0)

        # Identifier column for ED bed
        ed_units = bed_labels[bed_labels["ed"] == 1].bed_unit.tolist()
        df_beds["ed"] = np.where(df_beds["bed_unit"].isin(ed_units), 1, 0)

        # Identifier column for procedure bed
        procedure_units = bed_labels[bed_labels["procedure"] == 1].bed_unit.tolist()
        df_beds["procedure"] = np.where(df_beds["bed_unit"].isin(procedure_units), 1, 0)

        # Get rid of duplicate rows
        df_beds = (
            df_beds.groupby(["csn", "pat_id", "bed_location_start"])
            .first()
            .reset_index(level=(1, 2))
        )

        # drop unecessary cols
        df_beds = df_beds.drop(columns=drop_cols)

        self.df_beds = df_beds
        logging.info("Beds success")
###########################################################################
################### Create procedures df for all CSNs #####################
###########################################################################
    def import_procedures(self, drop_cols, index_col, date_cols):
        """
        Imports the procedures dataset from a CSV file, processes date columns, 
        drops unnecessary columns, and stores the cleaned DataFrame.

        Args:
            drop_cols (list of str): Columns to drop from the DataFrame.
            index_col (str): Column to use as the DataFrame index.
            date_cols (list of str): Columns to parse as datetime objects.
        """
        logging.info("Begin Procedures Import")
        
        path_procedures_file = self.file_dictionary["ORPROCEDURES"]

        # import procedure flat file and set date/time cols
        df_procedures = pd.read_csv(
            path_procedures_file,
            header=0,
            parse_dates=date_cols,
            infer_datetime_format=True,
            index_col=index_col,
            na_values=self.na_values,
            sep=self.delim,
            low_memory=False,
        )

        # drop unecessary cols
        df_procedures = df_procedures.drop(columns=drop_cols)

        self.df_procedures = df_procedures
        logging.info("Procedures success")
###########################################################################
################## Import ICD 9/10 Code df for all CSNs ###################
###########################################################################
    def import_diagnosis(self, drop_cols, index_col, date_cols):
        """
        Imports the diagnosis dataset, processes date columns, drops unnecessary columns, 
        and stores the cleaned DataFrame. Additionally, imports ICD10 comorbidity datasets 
        and stores them.

        Args:
            drop_cols (list of str): Columns to drop from the DataFrame.
            index_col (str or int): Column to use as the DataFrame index.
            date_cols (list of str): Columns to parse as datetime objects.
        """
        logging.info("Begin diagnosis import.")
        
        path_diagnosis_file = self.file_dictionary["DIAGNOSIS"]

        # import diganosis flat file
        df_diagnosis = pd.read_csv(
            path_diagnosis_file,
            header=0,
            parse_dates=date_cols,
            index_col=index_col,
            na_values=self.na_values,
            sep=self.delim,
            low_memory=False,
        )

        # drop unecessary cols
        df_diagnosis = df_diagnosis.drop(columns=drop_cols)

        self.df_diagnosis = df_diagnosis
        logging.info("Diagnosis success")
        
        logging.info("Begin comorbidity import.")

        #### ICD9 Portion
        # =============================================================================
        #         self.df_ahrq_ICD9 = self.make_comorbid_df(self.file_dictionary['ICD9_ahrq'],
        #                                         'ICD9',
        #                                         'ahrq',
        #                                         'dx_code_icd9',
        #                                         'v_ahrq_labels')
        #         # makes df for ahrq (like elix)
        #         self.df_elix_ICD9 = self.make_comorbid_df(self.file_dictionary['ICD9_elix'],
        #                                         'ICD9',
        #                                         'elix',
        #                                         'dx_code_icd9',
        #                                         'v_elix_labels')
        #         # makes df for Charlson
        #         self.df_quan_deyo_ICD9 = self.make_comorbid_df(self.file_dictionary['ICD9_quan_deyo'],
        #                                         'ICD9',
        #                                         'quan_deyo',
        #                                         'dx_code_icd9',
        #                                         'v_quan_deyo_labels')
        #         # makes df for Quan's Elix
        #         self.df_quan_elix_ICD9 = self.make_comorbid_df(self.file_dictionary['ICD9_quan_elix'],
        #                                 'ICD9',
        #                                 'quan_elix',
        #                                 'dx_code_icd9',
        #                                 'v_quan_elix_labels')
        #         # makes df for ccs
        #         self.df_ccs_ICD9 = self.make_comorbid_df(
        #                                 self.file_dictionary['ICD9_single_ccs'],
        #                                 'ICD9',
        #                                 'ccs_label',
        #                                 'dx_code_icd9',
        #                                 'v_ccs_labels')
        # =============================================================================

        #### ICD10 Portion
        # =============================================================================
        #         self.df_ahrq_ICD10 = self.make_comorbid_df(self.file_dictionary['ICD10_ahrq'],
        #                                 'ICD10',
        #                                 'ahrq',
        #                                 'dx_code_icd10',
        #                                 'v_ahrq_labels')
        #         # makes df for ahrq (like elix)
        #         self.df_elix_ICD10 = self.make_comorbid_df(self.file_dictionary['ICD10_elix'],
        #                                 'ICD10',
        #                                 'elix',
        #                                 'dx_code_icd10',
        #                                 'v_elix_labels')
        # =============================================================================
        # makes df for Charlson
        self.df_quan_deyo_ICD10 = self.make_comorbid_df(
            self.file_dictionary["ICD10_quan_deyo"],
            "ICD10",
            "quan_deyo",
            "dx_code_icd10",
            "v_quan_deyo_labels",
        )
        # makes df for Quan's Elix
        self.df_quan_elix_ICD10 = self.make_comorbid_df(
            self.file_dictionary["ICD10_quan_elix"],
            "ICD10",
            "quan_elix",
            "dx_code_icd10",
            "v_quan_elix_labels",
        )
        # =============================================================================
        #         # makes df for ccs
        #         self.df_ccs_ICD10 = self.make_comorbid_df(
        #                                 self.file_dictionary['ICD10_single_ccs'],
        #                                 'ICD10',
        #                                 'ccs_label',
        #                                 'dx_code_icd10',
        #                                 'v_ccs_labels')
        # =============================================================================
        logging.info("Comorbid success")

    def make_comorbid_df(
        self,
        comorbid_map_path,
        map_ICD_col,
        map_comorbidity_col,
        df_diagnosis_ICD_col,
        comorbid_labels,
    ):
        """
        Creates a comorbidity DataFrame by merging a mapping file with the diagnosis dataset, 
        and extracts the relevant comorbidity labels.

        Args:
            comorbid_map_path (str): Path to the CSV file containing the comorbidity mapping.
            map_ICD_col (str): Column name in the comorbidity map file containing the ICD code.
            map_comorbidity_col (str): Column name in the comorbidity map file containing the comorbidity type.
            df_diagnosis_ICD_col (str): Column name in the diagnosis DataFrame containing the ICD code.
            comorbid_labels (str): The name of the attribute where the list of comorbidity labels will be stored.

        Returns:
            pd.DataFrame: A DataFrame containing the merged data with patient IDs, diagnosis time, and comorbidity types.
        """
        # import mapping file
        map_df = pd.read_csv(comorbid_map_path, header=0)
        # column names in map file
        map_df_col_names = map_df.columns.to_list()

        # creates a variable with all the comorbidity types
        setattr(self, comorbid_labels, map_df[map_comorbidity_col].unique().tolist())

        # merge mapping file with diagnosis file
        all_diagnoses = self.df_diagnosis
        all_diagnoses[df_diagnosis_ICD_col] = (
            all_diagnoses[df_diagnosis_ICD_col]
            .astype(str)
            .str.replace(".", "", regex=False)
        )  # .drop_duplicates()
        mapped = map_df.merge(
            all_diagnoses.reset_index(),
            how="left",
            left_on=map_ICD_col,
            right_on=df_diagnosis_ICD_col,
        ).set_index("csn")
        return mapped[["pat_id", "dx_time_date"] + map_df_col_names]
