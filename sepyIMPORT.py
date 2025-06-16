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
import re
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
        # creates df with all vent mappings
        self.df_vent_mappings = pd.read_csv(file_dictionary["grouping_vent"])
        # creates df with all fluid groupings
        self.df_grouping_fluids = pd.read_csv(file_dictionary["grouping_fluids"])
    
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
        df = self._import_file_with_fallback(
            file_key, 
            utils.read_data_file,
            **params
        )

        # Convert all column names to lowercase for consistency
        if not df.empty:
            df.columns = df.columns.str.lower()

        return df

    def import_data(self, data_type: str, **kwargs) -> None:
        """
        Universal import function for all data types with flexible configuration.
        
        This function serves as a single entry point for importing any data type.
        It uses data type-specific processors to handle specialized processing after import.
        
        Args:
            data_type: Type of data to import (e.g., 'vitals', 'labs', 'vent')
            **kwargs: Optional parameters that override config defaults:
                      - file_key: Key to look up file path (defaults to uppercase data_type)
                      - index_col: Column(s) to use as the DataFrame index
                      - date_cols: Columns to parse as datetime objects
                      - drop_cols: Columns to drop from the DataFrame
                      - numeric_cols: Columns to convert to numeric values
                      - Any other parameters needed for processing
        """
        # Default file_key is uppercase version of data_type
        file_key = kwargs.pop('file_key', data_type.upper())
        
        # Common parameters for all imports
        common_params = {
            'file_key': file_key,
            'index_col': kwargs.pop('index_col', []),
            'date_cols': kwargs.pop('date_cols', None),
            'drop_cols': kwargs.pop('drop_cols', []),
            'numeric_cols': kwargs.pop('numeric_cols', []),
            'date_parser': kwargs.pop('date_parser', utils.d_parser),
            'memory_map': kwargs.pop('memory_map', True),
        }
        
        # Import the data using _common_import
        df = self._common_import(**common_params)
        
        # Store the dataframe with a consistent naming pattern
        df_attr_name = f"df_{data_type}"
        setattr(self, df_attr_name, df)
        
        # Call data type-specific processor if it exists
        processor_name = f"_process_{data_type}"
        logging.info(f"{data_type.capitalize()} import successful")
        if hasattr(self, processor_name) and callable(getattr(self, processor_name)):
            processor = getattr(self, processor_name)
            try:
                return_state = processor(df, **kwargs)
                if return_state == 1:
                    logging.info(f"{data_type.capitalize()} processing successful")
                else:
                    logging.warning(f"Processing for {data_type} may be incomplete")
            except Exception as e:
                logging.error(f"Error in {data_type} processing: {str(e)}")
            
    # Data-specific processors
    def _process_infusion_meds(self, df: pd.DataFrame, **kwargs) -> None:
        """Process infusion medications after initial import."""
        anti_infective_group_name = kwargs.get('anti_infective_group_name', 'anti_infective')
        vasopressor_group_name = kwargs.get('vasopressor_group_name', 'vasopressor')
        
        self.df_infusion_meds = df
        self._handle_duplicate_med_ids()
        
        # Process specific medication groups
        df_anti_infective_med_groups = self._filter_med_group(anti_infective_group_name)
        self.df_anti_infective_meds = (
            self.df_infusion_meds.reset_index()
            .merge(df_anti_infective_med_groups, how="inner", on="medication_id")
            .set_index("csn")
        )
        
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
        return 1 
    
    def _process_labs(self, df: pd.DataFrame, **kwargs) -> None:
        """Process labs after initial import."""
        def tidy_index(df: pd.DataFrame) -> pd.DataFrame:
            """
            Reorganize the multi-index DataFrame to a more usable format.
            """
            df = df.unstack(level=1)
            df.columns = df.columns.droplevel()
            df.columns.name = None
            df = df.droplevel(0)
            return df

        try:
            lab_groups = self.df_grouping_labs
            
            # Select Relevant Lab Groups
            lab_groups = lab_groups[["super_table_col_name", "component_id"]][
                lab_groups["import"] == "Yes"
            ]

            # Join Groups and Lab File
            df_labs_filtered = df.merge(lab_groups, how="inner", on="component_id")
            
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
            df_labs_filtered_string = df_labs_filtered.loc[
                df_labs_filtered.index.get_level_values("super_table_col_name").isin(
                    self.string_lab_col_names
                )
            ]
            
            # Select and Treat Labs that have Numeric value
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
            return 1
        except Exception as e:
            logging.error(f"Error processing labs: {str(e)}")
            # Create an empty DataFrame with the expected columns
            self.df_labs = pd.DataFrame(columns=self.all_lab_col_names)
            logging.warning("Using empty DataFrame for labs due to processing error")
            return 0
            
    def _process_vitals(self, df: pd.DataFrame, **kwargs) -> None:
        """Process vitals after initial import."""
        merge_cols = kwargs.get('merge_cols')
        
        # If there are columns to merge then do this:
        if merge_cols is not None:
            for merge_set in merge_cols:
                df[merge_set[2]] = df[merge_set[0]].fillna(df[merge_set[1]])
                df = df.drop(columns=[merge_set[0], merge_set[1]])

        # drop punctuation and make numeric
        df = utils.make_numeric(df, self.vital_col_names)
        self.df_vitals = df
        return 1

    def _process_vent(self, df: pd.DataFrame, **kwargs) -> None:
        """Process vent data after initial import."""
        # Map to vent_category
        vent_grouping = dict(zip(self.df_vent_mappings['vent_name'], self.df_vent_mappings['vent_cat']))
        
        # Apply the mapping to the dataframe
        df['vent_category'] = df['vent_mode'].map(vent_grouping)
        self.df_vent = df
        return 1

    def _process_bed_locations(self, df: pd.DataFrame, **kwargs) -> None:
        """Process bed locations after initial import."""
        bed_labels = self.df_bed_labels
        
        try:
            # drop anytimes where bed_location_start = bed_location_end
            df = df[df["bed_location_start"] != df["bed_location_end"]]

            # Identifier column for ICU bed
            icu_units = bed_labels[bed_labels["icu"] == 1].bed_units.tolist()
            df["icu"] = np.where(df["bed_unit"].isin(icu_units), 1, 0)

            # Identifier column for IMC
            imc_units = bed_labels[bed_labels["imc"] == 1].bed_units.tolist()
            df["imc"] = np.where(df["bed_unit"].isin(imc_units), 1, 0)

            # Identifier column for ED bed
            ed_units = bed_labels[bed_labels["ed"] == 1].bed_units.tolist()
            df["ed"] = np.where(df["bed_unit"].isin(ed_units), 1, 0)

            # Identifier column for procedure bed
            procedure_units = bed_labels[bed_labels["procedure"] == 1].bed_units.tolist()
            df["procedure"] = np.where(df["bed_unit"].isin(procedure_units), 1, 0)
            
            #Map bed_unit to icu_type and unit_type
            icu_type_mapping = dict(zip(bed_labels['bed_units'], bed_labels['icu_type']))
            unit_type_mapping = dict(zip(bed_labels['bed_units'], bed_labels['unit_type']))

            df['icu_type'] = df['bed_unit'].map(icu_type_mapping)
            df['unit_type'] = df['bed_unit'].map(unit_type_mapping)

            # Get rid of duplicate rows
            df = (
                df.groupby(["csn", "pat_id", "bed_location_start"])
                .first()
                .reset_index(level=(1, 2))
            )

            self.df_beds = df
            return 1
        except Exception as e:
            logging.error(f"Error processing bed locations: {str(e)}")
            # Create an empty DataFrame with expected columns
            self.df_beds = pd.DataFrame(columns=["bed_unit", "icu", "imc", "ed", "procedure", 
                                               "bed_location_start", "bed_location_end", "pat_id"])
            logging.warning("Using empty DataFrame for bed locations due to processing error")
            return 0

    def _process_gcs(self, df: pd.DataFrame, **kwargs) -> None:
        """Process GCS data after initial import."""
        try:
            # TODO: Why are we doing this? Doesn't cause any errors, but doesn't seem to be of any utility either.
            # merges all gcs values into a single timestamp/row
            df = df.groupby(["csn", "recorded_time"]).aggregate(
                {
                    "gcs_eye_score": ["mean"],
                    "gcs_verbal_score": ["mean"],
                    "gcs_motor_score": ["mean"],
                    "gcs_total_score": ["mean"],
                }
            )
            # drops the column index "mean" which came from agg fcn
            # also moves 'recorded time' out of index into a column
            df = df.droplevel(1, axis=1).reset_index(level="recorded_time")
            self.df_gcs = df
            return 1
        except Exception as e:
            logging.error(f"Error processing GCS data: {str(e)}")
            # Create an empty DataFrame with expected columns
            self.df_gcs = pd.DataFrame(columns=["recorded_time", "gcs_eye_score", "gcs_verbal_score", 
                                              "gcs_motor_score", "gcs_total_score"])
            logging.warning("Using empty DataFrame for GCS due to processing error")
            return 0
        
    def _process_diagnosis(self, df: pd.DataFrame, **kwargs) -> None:
        """Process diagnosis data after initial import."""
        self.df_diagnosis = df
        
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
            return 1
        except Exception as e:
            logging.error(f"Error processing comorbidity data: {str(e)}")
            # Create empty DataFrames with expected structure
            self.df_quan_deyo_ICD10 = pd.DataFrame(columns=["pat_id", "dx_time_date", "ICD10", "quan_deyo"])
            self.df_quan_elix_ICD10 = pd.DataFrame(columns=["pat_id", "dx_time_date", "ICD10", "quan_elix"])
            logging.warning("Using empty DataFrames for comorbidity data due to processing error")
            return 0

    def _process_radiology_notes(self, df: pd.DataFrame, **kwargs) -> None:
        """
        Process radiology notes after initial import.
        
        Handles text cleaning, tokenization, and any special processing needed for 
        radiology report text data.
        
        Args:
            df: DataFrame containing the imported radiology notes
            **kwargs: Optional parameters including:
                      - text_col: Name of the column containing the note text (default: 'report_text')
                      - clean_text: Whether to clean the text (default: True)
                      - max_length: Maximum text length to retain (default: None)
        """
        text_col = kwargs.get('text_col', 'report_text')
        clean_text = kwargs.get('clean_text', False)
        max_length = kwargs.get('max_length', None)
        
        try:
            # Check if text column exists
            if text_col not in df.columns:
                logging.warning(f"Text column '{text_col}' not found in radiology notes data")
                self.df_radiology_notes = df
                return
                
            # Basic text cleaning if requested
            if clean_text:
                # Convert to lowercase
                df[text_col] = df[text_col].str.lower()
                
                # Remove extra whitespace
                df[text_col] = df[text_col].str.replace(r'\s+', ' ', regex=True).str.strip()
                
                # Replace common abbreviations if needed
                # This can be expanded based on domain knowledge
                abbreviations = {
                    'w/': 'with ',
                    'w/o': 'without ',
                    'pt': 'patient ',
                    'hx': 'history '
                    # Add more abbreviations as needed
                }
                
                for abbr, replacement in abbreviations.items():
                    df[text_col] = df[text_col].str.replace(r'\b' + abbr + r'\b', replacement, regex=True)
            
            # Truncate long texts if max_length is specified
            if max_length:
                df[text_col] = df[text_col].str[:max_length]
            
            # Store the processed DataFrame
            self.df_radiology_notes = df
            return 1
        except Exception as e:
            logging.error(f"Error processing radiology notes: {str(e)}")
            # Create an empty DataFrame with expected structure
            self.df_radiology_notes = pd.DataFrame(columns=df.columns)
            logging.warning("Using empty DataFrame for radiology notes due to processing error")
            return 0
        
    def _process_clinical_notes(self, df: pd.DataFrame, **kwargs) -> None:
        """
        Process clinical notes after initial import.
        
        Handles text cleaning, section identification, and any special processing
        needed for clinical note text data.
        
        Args:
            df: DataFrame containing the imported clinical notes
            **kwargs: Optional parameters including:
                      - text_col: Name of the column containing the note text (default: 'note_text')
                      - clean_text: Whether to clean the text (default: True)
                      - extract_sections: Whether to extract common clinical note sections (default: False)
                      - max_length: Maximum text length to retain (default: None)
        """
        text_col = kwargs.get('text_col', 'note_text') 
        clean_text = kwargs.get('clean_text', False)
        extract_sections = kwargs.get('extract_sections', False)
        max_length = kwargs.get('max_length', None)
        
        try:
            # Check if text column exists
            if text_col not in df.columns:
                logging.warning(f"Text column '{text_col}' not found in clinical notes data")
                self.df_clinical_notes = df
                return
                
            # Basic text cleaning if requested
            if clean_text:
                # Convert to lowercase
                df[text_col] = df[text_col].str.lower()
                
                # Remove extra whitespace
                df[text_col] = df[text_col].str.replace(r'\s+', ' ', regex=True).str.strip()
                
                # Replace common abbreviations
                abbreviations = {
                    'w/': 'with ',
                    'w/o': 'without ',
                    'pt': 'patient ',
                    'hx': 'history ',
                    'dx': 'diagnosis ',
                    'rx': 'prescription ',
                    'tx': 'treatment '
                    # Add more abbreviations as needed
                }
                
                for abbr, replacement in abbreviations.items():
                    df[text_col] = df[text_col].str.replace(r'\b' + abbr + r'\b', replacement, regex=True)
            
            # Truncate long texts if max_length is specified
            if max_length:
                df[text_col] = df[text_col].str[:max_length]
            
            # Extract common clinical note sections if requested
            if extract_sections:
                # Define regex patterns for common sections
                section_patterns = {
                    'history': r'(?:history\s+of\s+present\s+illness|history|hpi)[\s\:]+(.+?)(?=\b(?:physical examination|assessment|plan|impression|vital signs|medications)\b|$)',
                    'physical_exam': r'(?:physical\s+examination|physical\s+exam)[\s\:]+(.+?)(?=\b(?:assessment|plan|impression|vital signs|medications)\b|$)',
                    'assessment': r'(?:assessment|impression)[\s\:]+(.+?)(?=\b(?:plan|recommendations|physical examination|medications)\b|$)',
                    'plan': r'(?:plan|recommendations)[\s\:]+(.+?)(?=\b(?:assessment|impression|physical examination|vital signs|medications)\b|$)'
                }
                
                # Extract each section
                for section_name, pattern in section_patterns.items():
                    df[f'section_{section_name}'] = df[text_col].str.extract(pattern, flags=re.IGNORECASE, expand=False)
            
            # Store the processed DataFrame
            self.df_clinical_notes = df
            return 1
        except Exception as e:
            logging.error(f"Error processing clinical notes: {str(e)}")
            # Create an empty DataFrame with expected structure
            self.df_clinical_notes = pd.DataFrame(columns=df.columns)
            logging.warning("Using empty DataFrame for clinical notes due to processing error")
            return 0

    def _process_in_out(self, df: pd.DataFrame, **kwargs) -> None:
        """Process in/out data after initial import."""
        self.df_in_out = df
        
        try:
            self.df_all_fluids = df
            self.df_all_fluids = self._process_in_out_data()
            
            # Filter ORDER_CATALOG_DESC to include only those in the df_grouping_fluids file with individual_fluid_import = 1 
            self.df_individual_bolus = self.df_all_fluids[
                self.df_all_fluids['ORDER_CATALOG_DESC'].isin(
                    self.df_grouping_fluids[self.df_grouping_fluids['individual_fluid_import'] == 1]['fluid_name']
                )
            ]
            return 1
        except Exception as e:
            logging.error(f"Error processing in/out data: {str(e)}")
            # Create an empty DataFrame with the expected columns
            self.df_all_fluids = pd.DataFrame(columns=df.columns.tolist() + ['volume'])
            self.df_individual_bolus = pd.DataFrame(columns=df.columns.tolist() + ['volume'])
            logging.warning("Using empty DataFrame for all fluids due to processing error")
            return 0

    def _process_in_out_data(self) -> pd.DataFrame:
        """
        Processes the 'ORDER_CLINICAL_DESC' column of a DataFrame to extract fluid intake.
        Returns processed fluid intake events DataFrame.
        """
        # Rename the method to avoid confusion with the processor function
        return self._process_in_out()
        
    # Keeping the original method signatures for backwards compatibility
    
    def import_encounters(self, drop_cols: List[str], index_col: str, date_cols: List[str]) -> None:
        """Legacy method for importing encounters. Use import_data('encounters', ...) instead."""
        self.import_data('encounters', drop_cols=drop_cols, index_col=index_col, date_cols=date_cols)

    def import_demographics(self, drop_cols: List[str], index_col: str, date_cols: List[str]) -> None:
        """Legacy method for importing demographics. Use import_data('demographics', ...) instead."""
        self.import_data('demographics', drop_cols=drop_cols, index_col=index_col, date_cols=date_cols)

    def import_infusion_meds(
        self,
        drop_cols: List[str],
        numeric_cols: List[str],
        anti_infective_group_name: str,
        vasopressor_group_name: str,
        index_col: str,
        date_cols: List[str],
    ) -> None:
        """Legacy method for importing infusion meds. Use import_data('infusion_meds', ...) instead."""
        self.import_data('infusion_meds', 
                       drop_cols=drop_cols, 
                       numeric_cols=numeric_cols,
                       anti_infective_group_name=anti_infective_group_name,
                       vasopressor_group_name=vasopressor_group_name,
                       index_col=index_col,
                       date_cols=date_cols)

    def import_labs(self, drop_cols: List[str], date_cols: List[str]) -> None:
        """Legacy method for importing labs. Use import_data('labs', ...) instead."""
        self.import_data('labs', drop_cols=drop_cols, date_cols=date_cols)

    def import_vitals(self, drop_cols: List[str], index_col: str, date_cols: List[str], merge_cols: Optional[List[List[str]]] = None) -> None:
        """Legacy method for importing vitals. Use import_data('vitals', ...) instead."""
        self.import_data('vitals', drop_cols=drop_cols, index_col=index_col, date_cols=date_cols, merge_cols=merge_cols)

    def import_vent(self, drop_cols: List[str], numeric_cols: List[str], index_col: str, date_cols: List[str]) -> None:
        """Legacy method for importing vent data. Use import_data('vent', ...) instead."""
        self.import_data('vent', drop_cols=drop_cols, numeric_cols=numeric_cols, index_col=index_col, date_cols=date_cols)
    
    def import_dialysis(self, drop_cols: List[str], numeric_cols: List[str], index_col: str, date_cols: List[str]) -> None:
        """Legacy method for importing dialysis data. Use import_data('dialysis', ...) instead."""
        self.import_data('dialysis', drop_cols=drop_cols, numeric_cols=numeric_cols, index_col=index_col, date_cols=date_cols)

    def import_in_out(self, drop_cols: List[str], numeric_cols: List[str], index_col: str, date_cols: List[str]) -> None:
        """Legacy method for importing in/out data. Use import_data('in_out', ...) instead."""
        self.import_data('in_out', drop_cols=drop_cols, numeric_cols=numeric_cols, index_col=index_col, date_cols=date_cols)

    def import_gcs(self, drop_cols: List[str], index_col: str, numeric_col: List[str], date_cols: List[str]) -> None:
        """Legacy method for importing GCS data. Use import_data('gcs', ...) instead."""
        self.import_data('gcs', drop_cols=drop_cols, index_col=index_col, numeric_cols=numeric_col, date_cols=date_cols)

    def import_cultures(self, drop_cols: List[str], index_col: str, date_cols: List[str]) -> None:
        """Legacy method for importing cultures data. Use import_data('cultures', ...) instead."""
        self.import_data('cultures', drop_cols=drop_cols, index_col=index_col, date_cols=date_cols)

    def import_bed_locations(self, drop_cols: List[str], index_col: str, date_cols: List[str]) -> None:
        """Legacy method for importing bed locations data. Use import_data('bed_locations', ...) instead."""
        self.import_data('bed_locations', drop_cols=drop_cols, index_col=index_col, date_cols=date_cols)

    def import_procedures(self, file_key: str, drop_cols: List[str], index_col: str, date_cols: List[str]) -> None:
        """Legacy method for importing procedures data. Use import_data('procedures', ...) instead."""
        self.import_data('procedures', file_key=file_key, drop_cols=drop_cols, index_col=index_col, date_cols=date_cols)

    def import_diagnosis(self, drop_cols: List[str], index_col: str, date_cols: List[str]) -> None:
        """Legacy method for importing diagnosis data. Use import_data('diagnosis', ...) instead."""
        self.import_data('diagnosis', drop_cols=drop_cols, index_col=index_col, date_cols=date_cols)

    def import_radiology_notes(self, drop_cols: List[str], index_col: str, date_cols: List[str], text_col: str = 'report_text', clean_text: bool = True, max_length: Optional[int] = None) -> None:
        """Legacy method for importing radiology notes. Use import_data('radiology_notes', ...) instead."""
        self.import_data('radiology_notes', 
                       drop_cols=drop_cols, 
                       index_col=index_col, 
                       date_cols=date_cols,
                       text_col=text_col,
                       clean_text=clean_text,
                       max_length=max_length)
    
    def import_clinical_notes(self, drop_cols: List[str], index_col: str, date_cols: List[str], text_col: str = 'note_text', clean_text: bool = True, extract_sections: bool = False, max_length: Optional[int] = None) -> None:
        """Legacy method for importing clinical notes. Use import_data('clinical_notes', ...) instead."""
        self.import_data('clinical_notes', 
                       drop_cols=drop_cols, 
                       index_col=index_col, 
                       date_cols=date_cols,
                       text_col=text_col,
                       clean_text=clean_text,
                       extract_sections=extract_sections,
                       max_length=max_length)

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
