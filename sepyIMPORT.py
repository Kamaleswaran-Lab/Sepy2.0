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


Changes Made by Mehak Arora:
    - Added import_dialysis and import_in_out functions
    - Created a utils function to read data files
    - Created init args for both sets of configs (import and data)
    - Removed delim as a function to SepyImport - was legacy, now makes zero sense to pass that as an argument
"""

import time
import utils
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Union, Optional, Any, Callable, TypeVar, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)

# Type variables for generic functions
T = TypeVar('T')
DataFrame = TypeVar('DataFrame', bound=pd.DataFrame)

@dataclass
class ImportConfig:
    """Configuration for sepyIMPORT class."""
    na_values: List[str]
    vital_col_names: List[str]
    vasopressor_units: List[str]
    numeric_lab_col_names: List[str]
    string_lab_col_names: List[str]
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        self.all_lab_col_names = self.numeric_lab_col_names + self.string_lab_col_names

@contextmanager
def timer(description: str) -> None:
    """
    Context manager for timing operations.
    
    Args:
        description: A description of the operation being timed.
    
    Yields:
        None
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    logging.info(f"{description} took {elapsed:.2f} seconds")

###########################################################################
############################### IMPORT Class ##############################
###########################################################################
class sepyIMPORT:
    """
    A class for importing and processing clinical data from CSV files, specifically designed
    to handle electronic medical records (EMR) datasets with various preprocessing steps.

    Args:
        file_dictionary: A dictionary containing file paths for various data files.
        sepyIMPORTConfigs: Configuration settings for data import.
        dataConfig: Configuration for data processing.
    """
    def __init__(self, file_dictionary: Dict[str, str], 
                 sepyIMPORTConfigs: Dict[str, Any]) -> None:
        # dictionary has file locations for flat files
        self.file_dictionary = file_dictionary

        # creates df with all medication groupings
        self.df_grouping_all_meds = pd.read_csv(file_dictionary["infusion_meds"])
        # creates df with all lab groupings
        self.df_grouping_labs = pd.read_csv(file_dictionary["grouping_labs"])
        # creates df with all bed location labels
        self.df_bed_labels = pd.read_csv(file_dictionary["bed_labels"])
        # creates df with all fluid groupings
        #self.df_grouping_fluids = pd.read_csv(file_dictionary["grouping_fluids"])
    
        # Create configuration object
        self.config = ImportConfig(
            na_values=sepyIMPORTConfigs["na_values"],
            vital_col_names=sepyIMPORTConfigs["vital_col_names"],
            vasopressor_units=sepyIMPORTConfigs["vasopressor_units"],
            numeric_lab_col_names=sepyIMPORTConfigs["numeric_lab_col_names"],
            string_lab_col_names=sepyIMPORTConfigs["string_lab_col_names"]
        )
        
        # For backward compatibility, maintain direct attributes
        self.na_values = self.config.na_values
        self.vital_col_names = self.config.vital_col_names
        self.vasopressor_units = self.config.vasopressor_units
        self.numeric_lab_col_names = self.config.numeric_lab_col_names
        self.string_lab_col_names = self.config.string_lab_col_names
        self.all_lab_col_names = self.config.all_lab_col_names
        
        logging.info("sepyIMPORT initialized")

    
    def _import_file_with_fallback(self, file_key: str, import_func: Callable, *args, **kwargs) -> pd.DataFrame:
        """
        Import a file with graceful fallback if the file is missing or corrupted.
        
        Args:
            file_key: The key for the file in the file_dictionary.
            import_func: The function to use for importing the file.
            *args: Positional arguments to pass to the import function.
            **kwargs: Keyword arguments to pass to the import function.
            
        Returns:
            DataFrame from successful import or empty DataFrame if import fails.
        """
        try:
            return import_func(*args, **kwargs)
        except FileNotFoundError:
            logging.warning(f"File {file_key} not found. Using empty DataFrame.")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error importing {file_key}: {str(e)}")
            return pd.DataFrame()
            
    def _common_import(self, 
                      file_key: str, 
                      index_col: Optional[Union[str, List[str]]] = None, 
                      date_cols: Optional[List[str]] = None, 
                      drop_cols: Optional[List[str]] = None, 
                      numeric_cols: Optional[List[str]] = None,
                      date_parser: Optional[Callable] = None,
                      memory_map: bool = True,
                      additional_params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Common logic for importing data files.
        
        Args:
            file_key: Key to look up the file path in file_dictionary.
            index_col: Column(s) to use as the DataFrame index.
            date_cols: Columns to parse as datetime objects.
            drop_cols: Columns to drop from the DataFrame.
            numeric_cols: Columns to convert to numeric values.
            date_parser: Function to use for parsing dates.
            memory_map: Whether to use memory mapping for file reading.
            additional_params: Additional parameters to pass to read_data_file.
            
        Returns:
            Imported and processed DataFrame.
        """
        params = {
            "file_path": self.file_dictionary.get(file_key, ""),
            "header": 0,
            "index_col": index_col,
            "date_cols": date_cols,
            "na_values": self.na_values,
            "memory_map": memory_map,
            "drop_cols": drop_cols,
            "low_memory": False,
        }
        
        if numeric_cols is not None:
            params["numeric_cols"] = numeric_cols
        
        if date_parser is not None:
            params["date_parser"] = date_parser
            
        if memory_map:
            params["memory_map"] = True
            
        if additional_params is not None:
            params.update(additional_params)
        
        # Use fallback mechanism for file import
        return self._import_file_with_fallback(
            file_key, 
            utils.read_data_file,
            **params
        )

    def import_encounters(self, drop_cols: List[str], index_col: str, date_cols: List[str]) -> None:
        """
        Imports the encounters dataset from a CSV file, parses date columns, 
        drops specified columns, and stores the cleaned DataFrame.

        Args:
            drop_cols: Columns to drop from the DataFrame.
            index_col: Column to use as the DataFrame index.
            date_cols: Columns to parse as datetime objects.
        """
        with timer("Importing encounters data"):
            logging.info("Begin Encounters Import")
            
            df_encounters = self._common_import(
                file_key="ENCOUNTER",
                index_col=index_col,
                date_cols=date_cols,
                drop_cols=drop_cols,
                date_parser=utils.d_parser
            )
            
            self.df_encounters = df_encounters
            logging.info("Encounters success")


    def import_demographics(self, drop_cols: List[str], index_col: str, date_cols: List[str]) -> None:
        """
        Imports the demographics dataset from a CSV file, parses date columns, 
        drops specified columns, and stores the cleaned DataFrame.

        Args:
            drop_cols: Columns to drop from the DataFrame.
            index_col: Column to use as the DataFrame index.
            date_cols: Columns to parse as datetime objects.
        """
        with timer("Importing demographics data"):
            logging.info("Begin Demographics Import")
            
            df_demographics = self._common_import(
                file_key="DEMOGRAPHICS",
                index_col=index_col,
                date_cols=date_cols,
                drop_cols=drop_cols,
                date_parser=utils.d_parser
            )
            
            self.df_demographics = df_demographics
            logging.info("Demographics success")

    def import_infusion_meds(
        self,
        drop_cols: List[str],
        numeric_cols: List[str],
        anti_infective_group_name: str,
        vasopressor_group_name: str,
        index_col: str,
        date_cols: List[str],
    ) -> None:
        """
        Imports the infusion medication dataset from a CSV file, processes the data by 
        converting specified columns to numeric, removes unnecessary columns, checks for 
        duplicates in medication IDs, and separates the data into anti-infective and 
        vasopressor medication groups.

        Args:
            drop_cols: Columns to drop from the DataFrame.
            numeric_cols: Columns to convert to numeric values.
            anti_infective_group_name: The name of the anti-infective medication group.
            vasopressor_group_name: The name of the vasopressor medication group.
            index_col: Column to use as the DataFrame index.
            date_cols: Columns to parse as datetime objects.
        """
        with timer("Importing and processing infusion medications"):
            # 1. Import the infusion medication data
            self._import_infusion_data(index_col, date_cols, numeric_cols, drop_cols)
            
            # 2. Check and handle duplicate medication IDs
            self._handle_duplicate_med_ids()
            
            # 3. Process specific medication groups
            self._process_anti_infective_meds(anti_infective_group_name)
            self._process_vasopressor_meds(vasopressor_group_name)

    def _import_infusion_data(self, 
                              index_col: str, 
                              date_cols: List[str], 
                              numeric_cols: List[str], 
                              drop_cols: List[str]) -> None:
        """
        Helper method to import infusion medication data from CSV.
        
        Args:
            index_col: Column to use as the DataFrame index.
            date_cols: Columns to parse as datetime objects.
            numeric_cols: Columns to convert to numeric values.
            drop_cols: Columns to drop from the DataFrame.
        """
        logging.info("Starting csv read for infusion meds.")
        
        self.df_infusion_meds = self._common_import(
            file_key="INFUSIONMEDS",
            index_col=index_col,
            date_cols=date_cols,
            numeric_cols=numeric_cols,
            drop_cols=drop_cols,
            date_parser=utils.d_parser,
            memory_map=True
        )

    def _handle_duplicate_med_ids(self) -> None:
        """
        Check for and handle duplicate medication IDs in the medication groupings.
        
        This function checks if there are duplicate medication_id entries in the
        df_grouping_all_meds DataFrame and removes them if found, keeping only the first
        occurrence of each medication_id.
        """
        rows_dropped = (
            self.df_grouping_all_meds.shape[0]
            - self.df_grouping_all_meds.drop_duplicates("medication_id").shape[0]
        )
        
        if rows_dropped > 0:
            self.df_grouping_all_meds = self.df_grouping_all_meds.drop_duplicates(
                subset="medication_id"
            )
            logging.info(
                f"You have {rows_dropped} duplicates of medication ID that were dropped!"
            )
        else:
            logging.info("Congrats, You have NO duplicates of medication ID!")

    def _filter_med_group(self, med_class_name: str) -> pd.DataFrame:
        """
        Filter medication groups by class name.
        
        Args:
            med_class_name: The medication class name to filter by.
            
        Returns:
            DataFrame containing filtered medication data.
        """
        return self.df_grouping_all_meds[
            self.df_grouping_all_meds["med_class"] == med_class_name
        ][["super_table_col_name", "medication_id"]]

    def _process_anti_infective_meds(self, anti_infective_group_name: str) -> None:
        """
        Process anti-infective medications.
        
        Args:
            anti_infective_group_name: The name of the anti-infective medication group.
        """
        df_anti_infective_med_groups = self._filter_med_group(anti_infective_group_name)

        self.df_anti_infective_meds = (
            self.df_infusion_meds.reset_index()
            .merge(df_anti_infective_med_groups, how="inner", on="medication_id")
            .set_index("csn")
        )

        logging.info("Anti-infective processing successful")

    def _process_vasopressor_meds(self, vasopressor_group_name: str) -> None:
        """
        Process vasopressor medications.
        
        Args:
            vasopressor_group_name: The name of the vasopressor medication group.
        """
        df_vasopressor_med_groups = self._filter_med_group(vasopressor_group_name)

        # Extract vasopressor medications and set multi-index
        df_vasopressor_meds = (
            self.df_infusion_meds.reset_index()
            .merge(df_vasopressor_med_groups, how="inner", on="medication_id")
            .reset_index()
            .set_index(["csn", "med_order_time", "super_table_col_name"], append=True)[
                ["med_action_dose", "med_action_dose_unit"]
            ]
        )

        # Process vasopressor data - unstack units and doses
        units = df_vasopressor_meds["med_action_dose_unit"].unstack(level=3)
        dose = df_vasopressor_meds["med_action_dose"].unstack(level=3)
        
        # Merge dose and units together
        df_vasopressor_meds = dose.merge(
            units, left_index=True, right_index=True, suffixes=["", "_dose_unit"]
        )

        # Clean up column naming and indexes
        df_vasopressor_meds.columns.name = None
        df_vasopressor_meds = df_vasopressor_meds.droplevel(0)
        df_vasopressor_meds = df_vasopressor_meds.reset_index(1)

        # Replace NaN values in unit columns
        df_vasopressor_meds[self.vasopressor_units] = df_vasopressor_meds[
            self.vasopressor_units
        ].replace({np.nan: ""})

        self.df_vasopressor_meds = df_vasopressor_meds
        logging.info("Vasopressor processing successful")

    def import_labs(self, 
                 drop_cols: List[str], 
                 date_cols: List[str] ) -> None:
        """
        Imports the lab dataset from a CSV file, processes both numeric and string lab values, 
        applies necessary transformations, and stores the cleaned DataFrame.

        Args:
            drop_cols: Columns to drop from the DataFrame.
            date_cols: Columns to parse as datetime objects.
        """
        ### The labs import needs a special function to tiddy the columns
        def tidy_index(df: pd.DataFrame) -> pd.DataFrame:
            """
            Reorganize the multi-index DataFrame to a more usable format.
            
            Args:
                df: DataFrame with multi-level index to be tidied.
                
            Returns:
                Tidied DataFrame with simplified structure.
            """
            # turns super_table_name into a col
            df = df.unstack(level=1)
            # unstack makes a multi-index for columns; this line removes "lab result" level
            df.columns = df.columns.droplevel()
            # this removes the "name" for all the columns i.e. super_table_col_names
            df.columns.name = None
            # removes numerical index for labs
            df = df.droplevel(0)
            return df

        with timer("Importing and processing lab data"):
            logging.info("Begin Lab Import")
            # Lab groups file has three important cols:
            # 1) component_id - the is the id number for the lab type
            # 2 )super_table_col_name- is the group name for similar labs
            # 3) physionet - indicates if used for physionet competition

            lab_groups = self.df_grouping_labs
            
            # Use the common import function with fallback
            df_labs = self._common_import(
                file_key="LABS",
                date_cols=date_cols,
                date_parser=utils.d_parser,
                drop_cols=drop_cols,
                memory_map=True
            )

            ### Select Relevant Lab Groups (Those set to import == Yes in the lab_groups file) ###
            lab_groups = lab_groups[["super_table_col_name", "component_id"]][
                lab_groups["import"] == "Yes"
            ]

            ### Join Groups and Lab File ###
            try:
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
                df_labs_filtered_numeric = tidy_index(df_labs_filtered_numeric)
                df_labs_filtered_string = tidy_index(df_labs_filtered_string)

                # Concat the string and numeric dfs
                df_labs = pd.concat([df_labs_filtered_numeric, df_labs_filtered_string], axis=0)
                
                # Remove duplicate columns
                df_labs = df_labs.loc[:, ~df_labs.columns.duplicated()]
            
                # if there are missing cols (i.e. no COVID in 2014) then it ensures the col name is added
                df_labs = df_labs.reindex(
                    df_labs.columns.union(self.all_lab_col_names), axis=1
                )
                
                self.df_labs = df_labs
                logging.info("Labs success")
            except Exception as e:
                logging.error(f"Error processing labs: {str(e)}")
                # Create an empty DataFrame with the expected columns
                self.df_labs = pd.DataFrame(columns=self.all_lab_col_names)
                logging.warning("Using empty DataFrame for labs due to processing error")

    def import_vitals(self, 
                  drop_cols: List[str], 
                  index_col: str, 
                  date_cols: List[str], 
                  merge_cols: Optional[List[List[str]]] = None) -> None:
        """
        Imports the vitals dataset from a CSV file, merges specified columns, 
        converts vital sign data to numeric, and stores the cleaned DataFrame.

        Args:
            drop_cols: Columns to drop from the DataFrame.
            index_col: Column to use as the DataFrame index.
            date_cols: Columns to parse as datetime objects.
            merge_cols: List of column groups to merge.
        """
        with timer("Importing and processing vitals data"):
            logging.info("Begin Vitals Import")

            # Use our common import with additional parameters for dtype
            additional_params = {"dtype": object}  # Use object dtype to prevent numeric conversion before specific processing
            
            df_vitals = self._common_import(
                file_key="VITALS",
                index_col=index_col,
                date_cols=date_cols,
                drop_cols=drop_cols,
                additional_params=additional_params
            )

            # If there are columns to merge then do this:
            if merge_cols is not None:
                for merge_set in merge_cols:
                    df_vitals[merge_set[2]] = df_vitals[merge_set[0]].fillna(
                        df_vitals[merge_set[1]]
                    )
                    df_vitals = df_vitals.drop(columns=[merge_set[0], merge_set[1]])

            # drop punctuation and make numeric
            df_vitals = utils.make_numeric(df_vitals, self.vital_col_names)

            self.df_vitals = df_vitals
            logging.info("Vitals success")

    def import_vent(self, 
                 drop_cols: List[str], 
                 numeric_cols: List[str], 
                 index_col: str, 
                 date_cols: List[str]) -> None:
        """
        Imports the ventilation dataset from a CSV file, processes numeric columns, 
        drops unnecessary columns, and stores the cleaned DataFrame.

        Args:
            drop_cols: Columns to drop from the DataFrame.
            numeric_cols: Columns to convert to numeric values.
            index_col: Column to use as the DataFrame index.
            date_cols: Columns to parse as datetime objects.
        """
        with timer("Importing vent data"):
            logging.info("Begin Vent Import")
            
            df_vent = self._common_import(
                file_key="VENT",
                index_col=index_col,
                date_cols=date_cols,
                numeric_cols=numeric_cols,
                drop_cols=drop_cols
            )

            self.df_vent = df_vent
            logging.info("Vent success")
    
    def import_dialysis(self, 
                       drop_cols: List[str], 
                       numeric_cols: List[str], 
                       index_col: str, 
                       date_cols: List[str]) -> None:
        """
        Imports the dialysis dataset from a CSV file, processes numeric columns, 
        drops unnecessary columns, and stores the cleaned DataFrame.
        
        Args:
            drop_cols: Columns to drop from the DataFrame.
            numeric_cols: Columns to convert to numeric values.
            index_col: Column to use as the DataFrame index.
            date_cols: Columns to parse as datetime objects.
        """
        with timer("Importing dialysis data"):
            logging.info("Begin Dialysis Import")
            
            df_dialysis = self._common_import(
                file_key="DIALYSIS",
                index_col=index_col,
                date_cols=date_cols,
                numeric_cols=numeric_cols,
                drop_cols=drop_cols
            )

            self.df_dialysis = df_dialysis
            logging.info("Dialysis success")

    def import_in_out(self, 
                     drop_cols: List[str], 
                     numeric_cols: List[str], 
                     index_col: str, 
                     date_cols: List[str]) -> None:
        """
        Imports the in/out dataset from a CSV file, processes numeric columns, 
        drops unnecessary columns, and stores the cleaned DataFrame.

        Args:
            drop_cols: Columns to drop from the DataFrame.
            numeric_cols: Columns to convert to numeric values.
            index_col: Column to use as the DataFrame index.
            date_cols: Columns to parse as datetime objects.
        """
        with timer("Importing in/out data"):
            logging.info("Begin In/Out Import")

            df_in_out = self._common_import(
                file_key="IN_OUT",
                index_col=index_col,
                date_cols=date_cols,
                numeric_cols=numeric_cols,
                drop_cols=drop_cols
            )

            self.df_in_out = df_in_out
            logging.info("In/Out success")

    def import_gcs(self, 
                drop_cols: List[str], 
                index_col: str, 
                numeric_col: List[str], 
                date_cols: List[str]) -> None:
        """
        Imports the GCS dataset from a CSV file, processes the numeric columns, 
        aggregates GCS scores by timestamp, and stores the cleaned DataFrame.

        Args:
            drop_cols: Columns to drop from the DataFrame.
            index_col: Column to use as the DataFrame index.
            numeric_col: Columns to convert to numeric values.
            date_cols: Columns to parse as datetime objects.
        """
        with timer("Importing GCS data"):
            logging.info("Begin GCS Import")

            df_gcs = self._common_import(
                file_key="GCS",
                index_col=index_col,
                date_cols=date_cols,
                numeric_cols=numeric_col,
                drop_cols=drop_cols
            )

            try:
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
            except Exception as e:
                logging.error(f"Error processing GCS data: {str(e)}")
                # Create an empty DataFrame with the expected columns
                self.df_gcs = pd.DataFrame(columns=["recorded_time", "gcs_eye_score", "gcs_verbal_score", 
                                                  "gcs_motor_score", "gcs_total_score"])
                logging.warning("Using empty DataFrame for GCS due to processing error")

    def import_cultures(self, 
                       drop_cols: List[str], 
                       index_col: str, 
                       date_cols: List[str]) -> None:
        """
        Imports the cultures dataset from a CSV file, processes date columns, 
        drops unnecessary columns, and stores the cleaned DataFrame.

        Args:
            drop_cols: Columns to drop from the DataFrame.
            index_col: Column to use as the DataFrame index.
            date_cols: Columns to parse as datetime objects.
        """
        with timer("Importing cultures data"):
            logging.info("Begin Cultures Import")

            df_cultures = self._common_import(
                file_key="CULTURES",
                index_col=index_col,
                date_cols=date_cols,
                drop_cols=drop_cols
            )

            self.df_cultures = df_cultures
            logging.info("Cultures success")

    def import_bed_locations(self, 
                           drop_cols: List[str], 
                           index_col: str, 
                           date_cols: List[str]) -> None:
        """
        Imports the bed locations dataset from a CSV file, processes date columns, 
        adds identifiers for various bed types (ICU, IMC, ED, procedure), 
        filters out duplicate rows, drops unnecessary columns, and stores the cleaned DataFrame.

        Args:
            drop_cols: Columns to drop from the DataFrame.
            index_col: Column to use as the DataFrame index.
            date_cols: Columns to parse as datetime objects.
        """
        with timer("Importing bed locations data"):
            logging.info("Begin Bed Import")

            bed_labels = self.df_bed_labels
            
            df_beds = self._common_import(
                file_key="BEDLOCATION",
                index_col=index_col,
                date_cols=date_cols
            )

            try:
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
                if drop_cols:
                    df_beds = df_beds.drop(columns=drop_cols)

                self.df_beds = df_beds
                logging.info("Beds success")
            except Exception as e:
                logging.error(f"Error processing bed locations: {str(e)}")
                # Create an empty DataFrame with expected columns
                self.df_beds = pd.DataFrame(columns=["bed_unit", "icu", "imc", "ed", "procedure", 
                                                   "bed_location_start", "bed_location_end", "pat_id"])
                logging.warning("Using empty DataFrame for bed locations due to processing error")

    def import_procedures(self, 
                        drop_cols: List[str], 
                        index_col: str, 
                        date_cols: List[str]) -> None:
        """
        Imports the procedures dataset from a CSV file, processes date columns, 
        drops unnecessary columns, and stores the cleaned DataFrame.

        Args:
            drop_cols: Columns to drop from the DataFrame.
            index_col: Column to use as the DataFrame index.
            date_cols: Columns to parse as datetime objects.
        """
        with timer("Importing procedures data"):
            logging.info("Begin Procedures Import")
            
            df_procedures = self._common_import(
                file_key="ORPROCEDURES",
                index_col=index_col,
                date_cols=date_cols,
                drop_cols=drop_cols
            )

            self.df_procedures = df_procedures
            logging.info("Procedures success")

    def import_diagnosis(self, 
                        drop_cols: List[str], 
                        index_col: str, 
                        date_cols: List[str]) -> None:
        """
        Imports the diagnosis dataset, processes date columns, drops unnecessary columns, 
        and stores the cleaned DataFrame. Additionally, imports ICD10 comorbidity datasets 
        and stores them.

        Args:
            drop_cols: Columns to drop from the DataFrame.
            index_col: Column to use as the DataFrame index.
            date_cols: Columns to parse as datetime objects.
        """
        with timer("Importing diagnosis and comorbidity data"):
            logging.info("Begin diagnosis import.")
            
            df_diagnosis = self._common_import(
                file_key="DIAGNOSIS",
                index_col=index_col,
                date_cols=date_cols,
                drop_cols=drop_cols
            )

            self.df_diagnosis = df_diagnosis
            logging.info("Diagnosis success")
            
            logging.info("Begin comorbidity import.")

            try:
                self.df_quan_deyo_ICD10 = self._make_comorbid_df(
                    self.file_dictionary["ICD10_quan_deyo"],
                    "ICD10",
                    "quan_deyo",
                    "dx_code_icd10",
                    "v_quan_deyo_labels",
                )
                # makes df for Quan's Elix
                self.df_quan_elix_ICD10 = self._make_comorbid_df(
                    self.file_dictionary["ICD10_quan_elix"],
                    "ICD10",
                    "quan_elix",
                    "dx_code_icd10",
                    "v_quan_elix_labels",
                )
                logging.info("Comorbid success")
            except Exception as e:
                logging.error(f"Error processing comorbidity data: {str(e)}")
                # Create empty DataFrames with expected structure
                self.df_quan_deyo_ICD10 = pd.DataFrame(columns=["pat_id", "dx_time_date", "ICD10", "quan_deyo"])
                self.df_quan_elix_ICD10 = pd.DataFrame(columns=["pat_id", "dx_time_date", "ICD10", "quan_elix"])
                logging.warning("Using empty DataFrames for comorbidity data due to processing error")

    def _make_comorbid_df(
        self,
        comorbid_map_path: str,
        map_ICD_col: str,
        map_comorbidity_col: str,
        df_diagnosis_ICD_col: str,
        comorbid_labels: str,
    ) -> pd.DataFrame:
        """
        Creates a comorbidity DataFrame by merging a mapping file with the diagnosis dataset, 
        and extracts the relevant comorbidity labels.

        Args:
            comorbid_map_path: Path to the CSV file containing the comorbidity mapping.
            map_ICD_col: Column name in the comorbidity map file containing the ICD code.
            map_comorbidity_col: Column name in the comorbidity map file containing the comorbidity type.
            df_diagnosis_ICD_col: Column name in the diagnosis DataFrame containing the ICD code.
            comorbid_labels: The name of the attribute where the list of comorbidity labels will be stored.

        Returns:
            A DataFrame containing the merged data with patient IDs, diagnosis time, and comorbidity types.
        """
        try:
            # import mapping file using the generic read_data_file function
            map_df = utils.read_data_file(
                file_path=comorbid_map_path,
                header=0
            )
            
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
            )
            mapped = map_df.merge(
                all_diagnoses.reset_index(),
                how="left",
                left_on=map_ICD_col,
                right_on=df_diagnosis_ICD_col,
            ).set_index("csn")
            return mapped[["pat_id", "dx_time_date"] + map_df_col_names]
        except Exception as e:
            logging.error(f"Error in _make_comorbid_df: {str(e)}")
            # Return an empty DataFrame with expected columns
            return pd.DataFrame(columns=["pat_id", "dx_time_date", map_ICD_col, map_comorbidity_col])
