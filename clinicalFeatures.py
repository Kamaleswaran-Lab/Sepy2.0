# -*- coding: utf-8 -*-
"""
clinicalFeatures.py

This module contains the refactored ClinicalDataProcessor and DerivedFeatures
classes that were previously embedded in *sepyDICT.py*.  Keeping these classes
in their own file improves modularity and makes them easier to maintain and
unit-test.

The implementation is a near verbatim extraction of the original code so that
behaviour remains unchanged.  Only the following *non-functional* tweaks were
introduced:

1.   Forward references to `SepyDictConfig` were replaced with the looser type
     `Any` in order to avoid circular imports.  The objects passed at runtime
     are still expected to comply with the same interface.
2.   All required imports were added explicitly so the module is completely
     self-contained.

No other logic was modified.
"""

from __future__ import annotations  # postpone evaluation of type hints

from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from typing import Any, Dict, List

import numpy as np
import pandas as pd

import utils
from comorbidipy import comorbidity


@dataclass
class SepyDictConfig:
    """Configuration class for sepyDICT with type safety and validation."""
    vital_col_names: List[str]
    numeric_lab_col_names: List[str]
    string_lab_col_names: List[str]
    gcs_col_names: List[str]
    bed_info: List[str]
    vasopressor_names: List[str]
    vasopressor_units: List[str]
    vasopressor_dose: List[str]
    individual_fluid_columns: List[str]
    lab_aggregation: Dict[str, str]
    constants: Dict[str, Any]
    quan_deyo_labels: List[str]
    quan_elix_labels: List[str]
    bounds: pd.DataFrame
    temperature_in_celsius: bool
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        self.all_lab_col_names = self.numeric_lab_col_names + self.string_lab_col_names
        self.vasopressor_col_names = self.vasopressor_names + self.vasopressor_units + self.vasopressor_dose
    
    @classmethod
    def initialize_config(cls, quan_deyo_labels: List[str], quan_elix_labels: List[str], bounds: pd.DataFrame, config_dict: Dict[str, Any]) -> 'SepyDictConfig':
        """Create configuration instance from dictionary."""
        # Get field names from the dataclass
        field_names = {field.name for field in cls.__dataclass_fields__.values()}
        
        # Filter config_dict to only include known fields
        filtered_config = {k: v for k, v in config_dict.items() if k in field_names}
        
        return cls(quan_deyo_labels=quan_deyo_labels, quan_elix_labels=quan_elix_labels, 
                   bounds = bounds, 
                   **filtered_config)
@dataclass
class ClinicalData:
    """Class to store clinical data for a patient."""
    pat_id: Any
    csn: Any
    config: SepyDictConfig
    beds: pd.DataFrame
    demographics: pd.DataFrame
    encounters: pd.DataFrame
    gcs: pd.DataFrame
    cultures: pd.DataFrame
    procedures: pd.DataFrame
    vent: pd.DataFrame
    diagnosis: pd.DataFrame
    labs: pd.DataFrame
    vasopressor_meds: pd.DataFrame
    anti_infective_meds: pd.DataFrame
    vitals: pd.DataFrame
    infusion_meds: pd.DataFrame
    quan_deyo_ICD10: pd.DataFrame
    quan_elix_ICD10: pd.DataFrame
    in_out_fluids: pd.DataFrame
    clinical_notes: pd.DataFrame
    radiology_notes: pd.DataFrame
    icd_procedures: pd.DataFrame
    cpt_procedures: pd.DataFrame
    dialysis: pd.DataFrame
    sirs_scores: pd.DataFrame | None = None
    sofa_scores: pd.DataFrame | None = None
    organ_system_scores: pd.DataFrame | None = None
    t_suspicion: pd.DataFrame | None = None
    t_sepsis3: pd.DataFrame | None = None
    
    #these are created in post_init
    #flags: Dict[str, Any]
    #super_table_time_index: pd.DatetimeIndex
    #event_times: Dict[str, Any]
    #static_features: Dict[str, Any]
    
    def __post_init__(self):
        """Validate and initialize after creation"""
        if not self.csn:
            raise ValueError("CSN must be provided")
        if not self.pat_id:
            raise ValueError("Patient ID must be provided")
        
        # Initialize flags
        self._initialize_flags()
        
        # Initialize event times if we have encounters data
        if self.encounters is not None:
            self._initialize_event_times()
            self._build_super_table_index()
            self._initialize_static_features()
            self.quan_deyo_ICD10_staging, self.quan_elix_ICD10_staging = pd.DataFrame(), pd.DataFrame()

    def _initialize_flags(self) -> None:
        """Initialize basic flags"""
        self.flags = {
            'csn': self.csn,
            'pat_id': self.pat_id,
            'y_vent_rows': 0,
            'y_vent_start_time': 0,
            'y_vent_end_time': 0,
            'vent_start_time': pd.NaT
        }

    def _initialize_event_times(self) -> None:
        """Initialize event times from encounters data"""
        def safe_extract(df: pd.DataFrame, column: str, default: Any = None) -> Any:
            try:
                return df.iloc[0][column] if not df.empty else default
            except (KeyError, IndexError):
                return default

        self.event_times = {
            'ed_presentation_time': safe_extract(self.encounters, 'ed_presentation_time'),
            'hospital_admission_date_time': safe_extract(self.encounters, 'hospital_admission_date_time'),
            'hospital_discharge_date_time': safe_extract(self.encounters, 'hospital_discharge_date_time')
        }
        
        # Calculate start index
        self.event_times['start_index'] = min(
            self.event_times['hospital_admission_date_time'],
            self.event_times['ed_presentation_time']
        )

        # Calculate ED wait time
        if all(time is not None for time in [
            self.event_times['hospital_admission_date_time'],
            self.event_times['ed_presentation_time']
        ]):
            self.flags['ed_wait_time'] = (
                self.event_times['hospital_admission_date_time'] - 
                self.event_times['ed_presentation_time']
            ).total_seconds() / 60
    
    def add_flag(self, name: str, value: Any) -> None:
        """Simple method to add or update a flag"""
        self.flags[name] = value
    
    def add_event_time(self, name: str, value: Any) -> None:
        """Simple method to add or update an event time"""
        self.event_times[name] = value
    
    def add_static_feature(self, name: str, value: Any) -> None:
        """Simple method to add or update a static feature"""
        self.static_features[name] = value

    def _build_super_table_index(self) -> None:
        """Build the timestamp index for the super_table"""
        if not all(key in self.event_times for key in ['start_index', 'hospital_discharge_date_time']):
            raise ValueError("Required event times not initialized")
            
        self.super_table_time_index = pd.date_range(
            self.event_times['start_index'],
            self.event_times['hospital_discharge_date_time'],
            freq=self.config.constants['resample_frequency']
        )

    def _initialize_static_features(self):

        #######################################
        # static_features: Patient demographic & encounter features that will not change during admisssion
        #######################################
        self.static_features = {}
        def safe_extract(df, column, default=None):
            try:
                return df.iloc[0, :][column] if not df.empty else default
            except (KeyError, IndexError):
                return default

        # Encounter features
        self.static_features['ed_arrival_source'] = safe_extract(self.encounters, 'ed_arrival_source')
        self.static_features['total_icu_days'] = safe_extract(self.encounters, 'total_icu_days', 0)
        self.static_features['discharge_status'] = safe_extract(self.encounters, 'discharge_status')
        self.static_features['discharge_to'] = safe_extract(self.encounters, 'discharge_to')
        self.static_features['encounter_type'] = safe_extract(self.encounters, 'encounter_type')
        self.static_features['age'] = safe_extract(self.encounters, 'age')
        self.static_features['admit_reason'] = safe_extract(self.encounters, 'admit_reason')

        # Demographics features
        self.static_features['gender'] = safe_extract(self.demographics, 'gender')
        self.static_features['gender_code'] = safe_extract(self.demographics, 'gender_code')
        self.static_features['race'] = safe_extract(self.demographics, 'race')
        self.static_features['race_code'] = safe_extract(self.demographics, 'race_code')
        self.static_features['ethnicity'] = safe_extract(self.demographics, 'ethnicity')
        self.static_features['ethnicity_code'] = safe_extract(self.demographics, 'ethnicity_code')
        

    @classmethod
    def from_sliced_data(cls, csn: Any, pat_id: Any, config: Any, sliced_data: Dict[str, pd.DataFrame]) -> ClinicalData:
        """Factory method to create from raw sliced data"""
        # Clean up dataframe names and create instance
        cleaned_data = {
            k.replace('_PerCSN', ''): v 
            for k, v in sliced_data.items()
        }
        return cls(csn=csn, pat_id=pat_id, config=config, **cleaned_data)

@dataclass
class supertable:
    """Class to store the supertable"""
    supertable: pd.DataFrame
    time_index: pd.DatetimeIndex

###############################################################################
# ClinicalDataProcessor
###############################################################################

class ClinicalDataProcessor:  
    """Handles data binning, cleaning, and aggregation operations."""

    def __init__(self, config: SepyDictConfig):
        self.config = config
        self.bounds = config.bounds 

        # Setup lab aggregation functions
        self.labAGG = self._setup_lab_aggregation()

        # Define categorical columns for memory optimisation
        self.categorical_columns = {
            "bed_unit": "category",
            "bed_type": "category",
            "icu_type": "category",
            "gender_code": "category",
            "vent_status": "int8",
            "on_vent": "int8",
            "on_pressors": "bool",
            "on_dobutamine": "bool",
            "on_dialysis": "int8",
            "history_of_dialysis": "int8",
            "infection": "int8",
            "sepsis": "int8",
        }

    # ---------------------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------------------

    def _setup_lab_aggregation(self) -> Dict[str, Any]:
        """Attach "smart" aggregation functions for the configured labs."""
        lab_agg: Dict[str, Any] = self.config.lab_aggregation.copy()
        for lab in lab_agg.keys():
            if len(self.bounds.loc[self.bounds["location in supertable"] == lab]) > 0:
                lab_agg[lab] = utils.agg_fn_wrapper(lab, self.bounds)
        return lab_agg
    
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Down-cast numeric dtypes and convert categoricals to save RAM."""
        df_optimized = df.copy()

        # Convert categorical columns first
        for col, dtype in self.categorical_columns.items():
            if col in df_optimized.columns:
                if dtype == "category":
                    df_optimized[col] = df_optimized[col].astype("category")
                elif dtype in ["int8", "bool"]:
                    df_optimized[col] = df_optimized[col].astype(dtype)

        # Optimise numeric columns
        numeric_cols = df_optimized.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in self.categorical_columns:
                if df_optimized[col].dtype in ["int64", "int32"]:
                    col_min = df_optimized[col].min()
                    col_max = df_optimized[col].max()

                    if col_min >= -128 and col_max <= 127:
                        df_optimized[col] = df_optimized[col].astype("int8")
                    elif col_min >= -32768 and col_max <= 32767:
                        df_optimized[col] = df_optimized[col].astype("int16")
                    elif col_min >= -2147483648 and col_max <= 2147483647:
                        df_optimized[col] = df_optimized[col].astype("int32")

                elif df_optimized[col].dtype == "float64":
                    df_optimized[col] = pd.to_numeric(df_optimized[col], downcast="float")

        return df_optimized

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def labs_staging(self, df: pd.DataFrame, time_index: pd.DatetimeIndex) -> None:
        """
        Resample lab values to config.constants['resample_frequency'] alignment.
        Apply the aggregation function defined in the config file.

        """
        if df.empty:
            df.index = df.index.get_level_values("collection_time")
            labs_df = pd.DataFrame(index=time_index, columns=df.columns)
            return labs_df
        else:
            df = df.reset_index("collection_time") #results in a multi-index df with lab supertable col name and collection_time as data and the rest as indices
            df = df.loc[:, ~df.columns.duplicated()] #removes duplicate columns
            resampled_data: Dict[str, pd.Series] = {}

            for key, agg_func in self.labAGG.items():
                if key in df.columns:
                    resampled_col = (
                        df[[key, "collection_time"]]
                        .set_index("collection_time")
                        .resample(self.config.constants['resample_frequency'], origin=time_index[0])
                        .apply(agg_func)
                        .reindex(time_index)
                    )
                    resampled_data[key] = resampled_col[key]

            labs_df = pd.DataFrame(resampled_data, index=time_index)
            labs_df = self.optimize_dataframe_memory(labs_df)
            return labs_df

    def vitals_staging(self, df: pd.DataFrame, time_index: pd.DatetimeIndex) -> None:  # noqa: C901 – complexity inherited
        """Resample vital signs."""
        if df.empty:
            vitals_df = pd.DataFrame(index=time_index, columns=df.columns)
            return vitals_df    

        resampled_data: Dict[str, pd.Series] = {}
        for key in self.config.vital_col_names:
            if key in df.columns:
                if len(self.bounds.loc[self.bounds["location in supertable"] == key]) > 0:
                    agg_fn = utils.agg_fn_wrapper(key, self.bounds)
                else:
                    agg_fn = "mean"

                resampled_col = (
                    df[[key, "recorded_time"]]
                    .set_index("recorded_time")
                    .resample(self.config.constants['resample_frequency'], origin=time_index[0])
                    .apply(agg_fn)
                    .reindex(time_index)
                )
                resampled_data[key] = resampled_col[key]

        vitals_df = pd.DataFrame(resampled_data, index=time_index)
        vitals_df = self.optimize_dataframe_memory(vitals_df)
        return vitals_df

    def gcs_staging(self, df: pd.DataFrame, time_index: pd.DatetimeIndex) -> None: 
        """Resample Glasgow Coma Scale values."""
        if df.empty:
            df = df.drop(columns=["recorded_time"])
            gcs_df = pd.DataFrame(index=time_index, columns=df.columns)
            return gcs_df

        new_df = pd.DataFrame()
        for key in self.config.gcs_col_names:
            if len(self.bounds.loc[self.bounds["location in supertable"] == key]) > 0:
                agg_fn = utils.agg_fn_wrapper_min(key, self.bounds)
            else:
                agg_fn = "min"
            col1 = (
                    df[[key, "recorded_time"]]
                    .resample(self.config.constants['resample_frequency'], on="recorded_time", origin=time_index[0])
                .apply(agg_fn)
            )
            new_df = pd.concat((new_df, col1), axis=1)
        gcs_df = new_df.reindex(time_index)
        return gcs_df
    
    def vent_staging(self, df: pd.DataFrame, time_index: pd.DatetimeIndex) -> None:
        """Resamples and aligns patient ventilator data to a unified hourly time index."""
        check_mech_vent_vars = ['vent_tidal_rate_set', 'peep']
        if df.empty:
            vent_df = pd.DataFrame(columns=['vent_status','vent_fio2', 'peep', 'vent_category', 'vent_tidal_rate_exhaled', 'vent_tidal_rate_set', 'vent_rate_set'], index=time_index)
            return vent_df
        else:
            vent_start = df[df.vent_start_time.notna()].vent_start_time.values
            vent_stop = df[df.vent_stop_time.notna()].vent_stop_time.values
            
            if len(vent_start) == 0: #CASE: Checking if there are valid mechanical ventilation rows that don't have a start time
                df['vent_status'] = np.where(df[check_mech_vent_vars].notnull().any(axis=1), 1, 0)
                
                if df['vent_status'].sum() > 0:
                    vent_start = df[df['vent_status'] > 0].recorded_time.iloc[0:1]
                else:
                    vent_start = []
                    
            #If there are valid mechanical ventilation rows, but no stop; add 6hrs to start time  
            if len(vent_start) != 0 and len(vent_stop) == 0:
                #flag identifies the presence of vent rows, and start time
                df['vent_status'] = np.where(df[check_mech_vent_vars].notnull().any(axis=1),1,0)
                #check if there are any "real" vent rows; if so 
                if df['vent_status'].sum()>0:
                    vent_stop  =  df[df['vent_status']>0].recorded_time.iloc[-1:]

            #Still no vent start after checking for valid mechanical ventilation rows
            if len(vent_start) == 0: #No valid mechanical ventilation values
                vent_status = pd.DataFrame(columns=['vent_status'], index=time_index) #all nans
            else:
                index = pd.Index([])
                vent_tuples = zip(vent_start, vent_stop )
    
                for pair in set(vent_tuples):
                    if pair[0] < pair[1]:
                        index = index.append( pd.date_range(pair[0], pair[1], freq=self.config.constants['resample_frequency']))
                    else: #In case of a mistake in start and stop recording
                        index = index.append( pd.date_range(pair[1], pair[0], freq=self.config.constants['resample_frequency']))  
                
                vent_status = pd.DataFrame(data=([1.0]*len(index)), columns =['vent_status'], index=index)
                
                #sets column to 1 if vent was on    
                vent_status = vent_status.resample(self.config.constants['resample_frequency'],
                                                   origin = time_index[0]).mean() \
                                                   .reindex(time_index)
        
        #Create vent_fio2
        agg_fn = utils.agg_fn_wrapper_max('fio2', self.bounds)
        vent_fio2 = df[['recorded_time','fio2']].resample(self.config.constants['resample_frequency'],
                                    on='recorded_time',
                                    origin=time_index[0]).apply(agg_fn) \
                                    .reindex(time_index)

        #Create peep
        agg_fn = utils.agg_fn_wrapper_max('peep', self.bounds)
        peep = df[['recorded_time','peep']].resample(self.config.constants['resample_frequency'],
                                    on='recorded_time',
                                    origin=time_index[0]).apply(agg_fn) \
                                    .reindex(time_index)

        #Create vent_category
        vent_category = df[['recorded_time','vent_category']].resample(self.config.constants['resample_frequency'],
                                    on='recorded_time',
                                    origin=time_index[0]).last() \
                                    .reindex(time_index)

        #Create vent_tidal_rate_exhaled
        agg_fn = utils.agg_fn_wrapper_max('vent_tidal_rate_exhaled', self.bounds)
        vent_tidal_rate_exhaled = df[['recorded_time','vent_tidal_rate_exhaled']].resample(self.config.constants['resample_frequency'],
                                    on='recorded_time',
                                    origin=time_index[0]).apply(agg_fn) \
                                    .reindex(time_index)

        #Create vent_tidal_rate_set
        agg_fn = utils.agg_fn_wrapper_max('vent_tidal_rate_set', self.bounds)
        vent_tidal_rate_set = df[['recorded_time','vent_tidal_rate_set']].resample(self.config.constants['resample_frequency'],
                                    on='recorded_time',
                                    origin=time_index[0]).apply(agg_fn) \
                                    .reindex(time_index)

        #Create vent_rate_set
        agg_fn = utils.agg_fn_wrapper_max('vent_rate_set', self.bounds)
        vent_rate_set = df[['recorded_time','vent_rate_set']].resample(self.config.constants['resample_frequency'],
                                    on='recorded_time',
                                    origin=time_index[0]).apply(agg_fn) \
                                    .reindex(time_index)

        vent_df = pd.DataFrame(index=time_index)
        vent_df['vent_status'] = vent_status
        vent_df['vent_fio2'] = vent_fio2
        vent_df['peep'] = peep
        vent_df['vent_category'] = vent_category
        vent_df['vent_tidal_rate_exhaled'] = vent_tidal_rate_exhaled
        vent_df['vent_tidal_rate_set'] = vent_tidal_rate_set
        vent_df['vent_rate_set'] = vent_rate_set
        return vent_df
        
    def procedures_staging(self, df: pd.DataFrame, time_index: pd.DatetimeIndex) -> None:
        def _create_procedure_series(procedures_df, index, time_col_start, time_col_end, procedure_name_col='primary_procedure_nm'):
            """
            Create a series aligned with supertable index showing procedure names
            for hours that match the specified time column
            """
            result_series = pd.Series(index=index, dtype='object')
            result_series[:] = np.nan
            
            for _, proc_row in procedures_df.iterrows():
                if pd.notna(proc_row[time_col_start]):
                    proc_hour = pd.Timestamp(proc_row[time_col_start]).floor(self.config.constants['resample_frequency'])

                    if proc_hour in index:
                        proc_hour_start = proc_hour
                    else:
                        logger.warning(f"Procedure {proc_row[procedure_name_col]} started at {proc_row[time_col_start]} but not found in supertable index")
                        continue 

                    if pd.notna(proc_row[time_col_end]):
                        proc_hour_end = pd.Timestamp(proc_row[time_col_end]).floor(self.config.constants['resample_frequency'])
                        if proc_hour_end in index and proc_hour_start <= proc_hour_end:
                            result_series.loc[proc_hour_start:proc_hour_end] = proc_row[procedure_name_col]
                        else:
                            logger.warning(f"Procedure {proc_row[procedure_name_col]} started at {proc_row[time_col_start]} but ended at {proc_row[time_col_end]}")
                            result_series.loc[proc_hour_start:] = proc_row[procedure_name_col]
                    else:
                        logger.warning(f"Procedure {proc_row[procedure_name_col]} started at {proc_row[time_col_start]} but {time_col_end} not found in supertable index")
                        result_series.loc[proc_hour_start:] = proc_row[procedure_name_col]
            
            return result_series
        
        if df.empty:
            procedures_df = pd.DataFrame(columns=['in_or', 'proc_start'], index=time_index)
            return procedures_df
        
        in_or_series = _create_procedure_series(df, time_index, 'in_or_dttm', 'out_or_dttm')
        proc_start_series = _create_procedure_series(df, time_index, 'procedure_start_dttm', 'procedure_comp_dttm')

        procedures_df = pd.DataFrame(index=time_index)
        procedures_df['in_or'] = in_or_series
        procedures_df['proc_start'] = proc_start_series
        return procedures_df
    
    def icd_staging(self, df: pd.DataFrame, time_index: pd.DatetimeIndex) -> None:
        if df.empty:
            icd_df = pd.DataFrame(columns=['icd_procedure_desc', 'icd10_procedure_code'], index=time_index)
            return icd_df

        df['procedure_date'] = pd.to_datetime(df['procedure_date'])
        df = df.set_index("procedure_date")
        # Aggregate procedure_desc with "\n", icd10_procedure_code with ";"
        desc_agg = df['procedure_desc'].groupby(
            pd.Grouper(freq=self.config.constants['resample_frequency'], origin=time_index[0])
        ).apply(lambda x: "\n".join(x.dropna().astype(str)) if not x.isna().all() else "")

        code_agg = df['icd10_procedure_code'].groupby(
            pd.Grouper(freq=self.config.constants['resample_frequency'], origin=time_index[0])
        ).apply(lambda x: ";".join(x.dropna().astype(str)) if not x.isna().all() else "")
        
        icd_df = pd.DataFrame({
            'icd_procedure_desc': desc_agg.reindex(time_index, fill_value=""),
            'icd10_procedure_code': code_agg.reindex(time_index, fill_value="")
        }, index=time_index)
        return icd_df

    def cpt_staging(self, df: pd.DataFrame, time_index: pd.DatetimeIndex) -> None:
        if df.empty:
            cpt_df = pd.DataFrame(columns=['cpt_procedure_desc', 'cpt_procedure_cd'], index=time_index)
            return cpt_df

        df['procedure_dttm'] = pd.to_datetime(df['procedure_dttm'])
        df = df.set_index("procedure_dttm")
        # Aggregate procedure_cpt_desc with "\n", procedure_cpt_cd with ";"
        desc_agg = df['procedure_cpt_desc'].groupby(
            pd.Grouper(freq=self.config.constants['resample_frequency'], origin=time_index[0])
        ).apply(lambda x: "\n".join(x.dropna().astype(str)) if not x.isna().all() else "")

        code_agg = df['procedure_cpt_cd'].groupby(
            pd.Grouper(freq=self.config.constants['resample_frequency'], origin=time_index[0])
        ).apply(lambda x: ";".join(x.dropna().astype(str)) if not x.isna().all() else "")
        
        cpt_df = pd.DataFrame({
            'cpt_procedure_desc': desc_agg.reindex(time_index, fill_value=""),
            'cpt_procedure_cd': code_agg.reindex(time_index, fill_value="")
        }, index=time_index)
        return cpt_df

    def clinical_notes_staging(self, df: pd.DataFrame, time_index: pd.DatetimeIndex) -> None:
        if df.empty:
            clinical_notes_df = pd.DataFrame(columns=['clinical_notes'], index=time_index)
            return clinical_notes_df

        clinical_notes_df = df.resample(self.config.constants['resample_frequency'], origin=time_index[0]).last()
        clinical_notes_df = clinical_notes_df.reindex(time_index)
        return clinical_notes_df

    def radiology_notes_staging(self, df: pd.DataFrame, time_index: pd.DatetimeIndex) -> None:
        if df.empty:
            radiology_notes_df = pd.DataFrame(columns=['radiology_notes'], index=time_index)
            return radiology_notes_df

        df['day_verified'] = pd.to_datetime(df['day_verified'])
        df = df.set_index("day_verified")
        notes_acc_nbr = df['acc_nbr'].groupby(
            pd.Grouper(freq=self.config.constants['resample_frequency'], origin=time_index[0])
        ).apply(lambda x: ','.join(x.astype(str)))

        radiology_notes_df = notes_acc_nbr.reindex(time_index, fill_value=0)
        radiology_notes_df = pd.DataFrame(radiology_notes_df, columns = ['radiology_acc_nbr'])
        return radiology_notes_df
    
    def vasopressor_staging(self, df: pd.DataFrame, time_index: pd.DatetimeIndex) -> None:
        vas_cols = self.config.vasopressor_names + self.config.vasopressor_units + ['med_order_time']
        df =df[vas_cols]
        vas_keys = self.config.vasopressor_names + self.config.vasopressor_units
        
        if df.empty:
            df = df.drop(columns=['med_order_time'])
            vasopressor_meds_df = pd.DataFrame(columns = df.columns, index = time_index)
        else:
            new = pd.DataFrame([])
            for key in vas_keys:
                if len(self.bounds.loc[self.bounds['location in supertable'] == key]) > 0:
                    agg_fn = utils.agg_fn_wrapper_max(key, self.bounds)
                else:
                    agg_fn = "max"
                col1 = df[[key, 'med_order_time']].resample('60min', on = "med_order_time",  \
                                                           origin = time_index[0]).apply(agg_fn)
                #col1 = col1.drop(columns=['med_order_time'])

                new = pd.concat((new, col1), axis = 1)
            vasopressor_meds_df = new.reindex(time_index)
            
        return vasopressor_meds_df

    def individual_fluids_staging(self, df_in_out_fluids: pd.DataFrame, time_index: pd.DatetimeIndex) -> None:
        if df_in_out_fluids.empty:
            individual_fluids_df = pd.DataFrame(columns=self.config.individual_fluid_columns, index=time_index)
            return individual_fluids_df
        
        volume_cols = [x for x in df_in_out_fluids.columns if x.endswith("_volume")]
        df_in_out_fluids = df_in_out_fluids[volume_cols]
        df_in_out_fluids.columns = [x.replace("_volume", "") for x in df_in_out_fluids.columns]
        df_in_out_fluids = df_in_out_fluids.reset_index("service_ts") 
        resampled_data: Dict[str, pd.Series] = {}

        for column in self.config.individual_fluid_columns:
            resampled_col = (
                df_in_out_fluids[[column, "service_ts"]]
                .set_index("service_ts")
                .resample(self.config.constants['resample_frequency'], origin=time_index[0])
                .last()
                .reindex(time_index)
            )
            resampled_data[column] = resampled_col[column]
            
            # if values per hour are less than 250ml per hour, set to 0
            # TODO: this is such a hack, we should have a better way to do this
            if self.config.constants['resample_frequency'] == '1H' or self.config.constants['resample_frequency'] == '60min':
                threshold = 250
            elif self.config.constants['resample_frequency'] == '15min':
                threshold = 250.0/4
            elif self.config.constants['resample_frequency'] == '5min':
                threshold = 250.0/12
            elif self.config.constants['resample_frequency'] == '30min':
                threshold = 250.0/2
            else:
                threshold = 0
            resampled_col[column] = resampled_col[column].apply(lambda x: 0 if x < threshold else x) #if values per hour are less than 250ml per hour, set to 0

        individual_fluids_df = pd.DataFrame(resampled_data, index=time_index)
        
        return individual_fluids_df
        
    def cumulative_fluids_staging(self, df_in_out_fluids: pd.DataFrame, df_infusion_meds: pd.DataFrame, time_index: pd.DatetimeIndex) -> None:
        if df_in_out_fluids.empty and df_infusion_meds.empty:
            cumulative_fluids_df = pd.DataFrame(columns=['cumulative_fluids'], index=time_index)
            return cumulative_fluids_df
        
        ## Fluids from in_out_fluids
        volume_cols = [x for x in df_in_out_fluids.columns if x.endswith("_volume")]
        df_in_out_fluids = df_in_out_fluids[volume_cols]
        df_in_out_fluids.columns = [x.replace("_volume", "") for x in df_in_out_fluids.columns]
        df_in_out_fluids = df_in_out_fluids.reset_index("service_ts") 

        resampled_data: Dict[str, pd.Series] = {}
        all_cols = df_in_out_fluids.columns.tolist()
        all_cols.remove('service_ts')
        for column in all_cols:
            resampled_col = (
                df_in_out_fluids[[column, "service_ts"]]
                .set_index("service_ts")
                .resample(self.config.constants['resample_frequency'], origin=time_index[0])
                .last()
                .reindex(time_index)
            )
            resampled_data[column] = resampled_col[column]

        all_fluids_df = pd.DataFrame(resampled_data, index=time_index)
        cumulative_fluids_df = pd.DataFrame(index=time_index, columns = ['cumulative_fluids'])
        cumulative_fluids_df['cumulative_fluids'] = all_fluids_df.sum(axis=1)
        
        #Fluids from infusion meds
        df_infusion_meds = df_infusion_meds.loc[~df_infusion_meds.med_name.isin(self.config.individual_fluid_columns)]
        df_infusion_meds = df_infusion_meds.loc[df_infusion_meds.volume != "none"]
        df_infusion_meds.set_index("med_action_time", inplace=True)
        df_infusion_meds_volume = df_infusion_meds['volume'].resample(self.config.constants['resample_frequency'], origin=time_index[0]).sum()
        df_infusion_meds_volume = df_infusion_meds_volume.reindex(time_index).fillna(0)
        
        cumulative_fluids_df['all_fluids'] = cumulative_fluids_df['cumulative_fluids'] + df_infusion_meds_volume
        cumulative_fluids_df['cumulative_fluids'] = cumulative_fluids_df['all_fluids'].cumsum()
        return cumulative_fluids_df
        
    
    def static_features_staging(self, age: int, gender: Any, diagnosis_PerCSN: pd.DataFrame, time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Creates columns for age, gender, and comorbidity scores
        Args:
            age: int
            gender: Any
            diagnosis_PerCSN: pd.DataFrame
            time_index: pd.DatetimeIndex
        Returns:
            pd.DataFrame
        """
        #Get static features
        df = pd.DataFrame()
        df['code'] = diagnosis_PerCSN['dx_code_icd9'].values
        df['age'] = [age]*len(df)
        df['id'] = diagnosis_PerCSN.index

        if all(df['code'] == '--') or pd.isnull(df['code']).all():
            cci9 = None
        else:
            df_out = comorbidity(df,  
                                 id="id",
                                 code="code",
                                 age="age",
                                 score="charlson",
                                 icd="icd9",
                                 variant="quan",
                                 assign0=True)
            cci9 = df_out['comorbidity_score'].values[0]

        df = pd.DataFrame()
        df['code'] = diagnosis_PerCSN['dx_code_icd10'].values
        df['age'] = [age]*len(df)
        df['id'] = diagnosis_PerCSN.index

        if all(df['code'] == '--') or pd.isnull(df['code']).all():
            cci10 = None
        else:
            df_out = comorbidity(df,  
                                 id="id",
                                 code="code",
                                 age="age",
                                 score="charlson",
                                 icd="icd10",
                                 variant="shmi",
                                 weighting="shmi",
                                 assign0=True)
            cci10 = df_out['comorbidity_score'].values[0]
        
        static_features = pd.DataFrame(index=time_index)
        static_features['age'] = [age]*len(static_features)
        static_features['gender'] = [gender]*len(static_features)
        static_features['cci9'] = [cci9]*len(static_features)
        static_features['cci10'] = [cci10]*len(static_features)

        return static_features

    
    def comorbidity_staging(self, df_quan_deyo: pd.DataFrame, df_quan_elix: pd.DataFrame) -> None:                        
        quan_deyo_ICD10_staging = df_quan_deyo.reset_index().groupby(['ICD10']).first().\
                                groupby(['quan_deyo']).agg(
                                icd_count = pd.NamedAgg(column="csn", aggfunc="count"),
                                date_time = pd.NamedAgg(column="dx_time_date", aggfunc="first"))\
                                .reindex(self.config.quan_deyo_labels).rename_axis(None)
                                #.agg({'csn':'count', 'dx_time_date':'first'})\

                                
        quan_elix_ICD10_staging = df_quan_elix.reset_index().groupby(['ICD10']).first().\
                                groupby(['quan_elix']).agg(
                                icd_count = pd.NamedAgg(column="csn", aggfunc="count"),
                                date_time = pd.NamedAgg(column="dx_time_date", aggfunc="first"))\
                                .reindex(self.config.quan_elix_labels).rename_axis(None)
                                #.agg({'csn':'count', 'dx_time_date':'first'})\
        return quan_deyo_ICD10_staging, quan_elix_ICD10_staging
    
    def assign_bed_status(self, df: pd.DataFrame, time_index: pd.DatetimeIndex) -> None:
        #these columns have the flags for bed status
        bed_category_names = ["icu", "imc", "ed", "procedure"]
        #makes an empty dataframe
        bed_status = pd.DataFrame(columns = bed_category_names)
        
        for i, row in df.iterrows():
            #makes an hourly index from bed strat to bed end
            index = pd.date_range(row['bed_location_start'], row['bed_location_end'], freq=self.config.constants['resample_frequency'])
            
            #makes a df for a single bed with the index and bed category values
            single_bed = pd.DataFrame(data = np.repeat([row[bed_category_names].values], len(index), axis=0),    
                                      columns = bed_category_names,
                                      index = index)
            #adds all beds to single df
            bed_status = pd.concat([bed_status, single_bed])  
        bed_status = bed_status[~bed_status.index.duplicated(keep='first')]
        
        #this is bed status re_indexed with super_table index; gets merged in later
        bed_status = bed_status.reindex(time_index, method='nearest')
        return bed_status

    def create_bed_unit(self, df: pd.DataFrame, time_index: pd.DatetimeIndex) -> None:
        """Create bed unit and related columns."""
            
        bed_start = df['bed_location_start'].values
        bed_end = df['bed_location_end'].values
        bed_unit = df['bed_unit'].values
        bed_type = df['unit_type'].values
        icu_type = df['icu_type'].values

        bed_status = pd.DataFrame(index=time_index)
        bed_status['bed_unit'] = [0]*len(time_index)
        bed_status['bed_type'] = [0]*len(time_index)
        bed_status['icu_type'] = [0]*len(time_index)

        for i in range(len(df)):
            start = bed_start[i]
            end = bed_end[i]
            unit = bed_unit[i]
            idx = np.bitwise_and(time_index >= start ,  time_index <= end)
            bed_status.loc[idx, 'bed_unit'] = unit
            bed_status.loc[idx, 'bed_type'] = bed_type[i]
            bed_status.loc[idx, 'icu_type'] = icu_type[i]

        return bed_status
    
    def dialysis_staging(self, df: pd.DataFrame, time_index: pd.DatetimeIndex) -> None:
        """Create dialysis status column."""
        if df.empty:
            on_dialysis_df = pd.DataFrame(columns=['on_dialysis'], index=time_index)
            return on_dialysis_df
        
        on_dialysis_df = pd.DataFrame(index=time_index)
        on_dialysis_df['on_dialysis'] = [0]*len(time_index)
        for time in df['service_timestamp']:
            time = pd.to_datetime(time)
            on_dialysis_df.loc[(on_dialysis_df.index - time > pd.Timedelta('0 seconds')), 'on_dialysis'] = 1
        return on_dialysis_df
    
    def create_history_of_dialysis(self, diagnosis: pd.DataFrame, time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Create history of dialysis indicator column."""
        icd9_code = '585.6'
        icd10_code = 'N18.6'
        if diagnosis.empty:
            history_of_dialysis = [0]*len(time_index)
        else:
            check = any(diagnosis.dx_code_icd9 == icd9_code) | any(diagnosis.dx_code_icd10 == icd10_code)
            history_of_dialysis = [1 if check else 0]*len(time_index)
        return pd.DataFrame(history_of_dialysis, index=time_index, columns=['history_of_dialysis'], dtype='int32')
    
    def process_clinical_data(self, clinical_data: ClinicalData) -> None:
        """Process clinical data and create supertable"""
        
        #Step 1: Initialize supertable
        supertable_df = supertable(supertable=pd.DataFrame(index=clinical_data.super_table_time_index), time_index=clinical_data.super_table_time_index)
        logging.info(f"Supertable data class object created with {len(supertable_df.supertable)} rows.")
        
        #Step 2: Add static features
        static_features_columns = self.static_features_staging(clinical_data.static_features['age'],\
                                                             clinical_data.static_features['gender'],
                                                             clinical_data.diagnosis,
                                                             supertable_df.time_index)
        supertable_df.supertable = pd.concat([supertable_df.supertable, static_features_columns], axis=1)
        logging.info(f"Static features added to supertable.")
        
        #Step 3: Add labs
        labs_columns = self.labs_staging(clinical_data.labs, supertable_df.time_index)
        supertable_df.supertable = pd.concat([supertable_df.supertable, labs_columns], axis=1)
        logging.info(f"Labs added to supertable.")

        #Step 4: Add vitals
        vitals_columns = self.vitals_staging(clinical_data.vitals, supertable_df.time_index)
        supertable_df.supertable = pd.concat([supertable_df.supertable, vitals_columns], axis=1)
        logging.info(f"Vitals added to supertable.")
        
        #Step 5: Add procedures
        procedures_columns = self.procedures_staging(clinical_data.procedures, supertable_df.time_index)
        supertable_df.supertable = pd.concat([supertable_df.supertable, procedures_columns], axis=1)
        logging.info(f"Procedures added to supertable.")

        #Step 6: gcs staging
        gcs_columns = self.gcs_staging(clinical_data.gcs, supertable_df.time_index)
        supertable_df.supertable = pd.concat([supertable_df.supertable, gcs_columns], axis=1)
        logging.info(f"GCS added to supertable.")

        #Step 7: Add vent status
        vent_columns = self.vent_staging(clinical_data.vent, supertable_df.time_index)
        supertable_df.supertable = pd.concat([supertable_df.supertable, vent_columns], axis=1)
        logging.info(f"Vent status added to supertable.")

        #Step 8: Add bed status
        bed_status_columns = self.assign_bed_status(clinical_data.beds, supertable_df.time_index)
        supertable_df.supertable = pd.concat([supertable_df.supertable, bed_status_columns], axis=1)
        logging.info(f"Bed status added to supertable.")

        #Step 9: Add bed unit
        bed_unit_columns = self.create_bed_unit(clinical_data.beds, supertable_df.time_index)
        supertable_df.supertable = pd.concat([supertable_df.supertable, bed_unit_columns], axis=1)
        logging.info(f"Bed unit added to supertable.")

        #Step 10: Add dialysis status
        dialysis_columns = self.dialysis_staging(clinical_data.dialysis, supertable_df.time_index)
        supertable_df.supertable = pd.concat([supertable_df.supertable, dialysis_columns], axis=1)
        logging.info(f"Dialysis status added to supertable.")

        #Step 11: Add history of dialysis
        history_of_dialysis_columns = self.create_history_of_dialysis(clinical_data.diagnosis, supertable_df.time_index)
        supertable_df.supertable = pd.concat([supertable_df.supertable, history_of_dialysis_columns], axis=1)
        logging.info(f"History of dialysis added to supertable.")

        #Step 12: Add fluids
        individual_fluids_columns = self.individual_fluids_staging(clinical_data.in_out_fluids, supertable_df.time_index)
        supertable_df.supertable = pd.concat([supertable_df.supertable, individual_fluids_columns], axis=1)
        logging.info(f"Individual fluids added to supertable.")

        #Step 13: Add cumulative fluids
        cumulative_fluids_columns = self.cumulative_fluids_staging(clinical_data.in_out_fluids, clinical_data.infusion_meds, supertable_df.time_index)
        supertable_df.supertable = pd.concat([supertable_df.supertable, cumulative_fluids_columns], axis=1)
        logging.info(f"Cumulative fluids added to supertable.")

        #Step 14: radiology notes
        radiology_notes_columns = self.radiology_notes_staging(clinical_data.radiology_notes, supertable_df.time_index)
        supertable_df.supertable = pd.concat([supertable_df.supertable, radiology_notes_columns], axis=1)
        logging.info(f"Radiology notes added to supertable.")

        #Step 15: clinical notes
        clinical_notes_columns = self.clinical_notes_staging(clinical_data.clinical_notes, supertable_df.time_index)
        supertable_df.supertable = pd.concat([supertable_df.supertable, clinical_notes_columns], axis=1)
        logging.info(f"Clinical notes added to supertable.")
        
        #Step 16: Add vasopressor meds
        vasopressor_meds_columns = self.vasopressor_staging(clinical_data.vasopressor_meds, supertable_df.time_index)
        supertable_df.supertable = pd.concat([supertable_df.supertable, vasopressor_meds_columns], axis=1)
        logging.info(f"Vasopressor meds added to supertable.")

        #Step 18: Add icd procedures
        icd_procedures_columns = self.icd_staging(clinical_data.icd_procedures, supertable_df.time_index)
        supertable_df.supertable = pd.concat([supertable_df.supertable, icd_procedures_columns], axis=1)
        logging.info(f"ICD procedures added to supertable.")

        #Step 19: Add cpt procedures
        cpt_procedures_columns = self.cpt_staging(clinical_data.cpt_procedures, supertable_df.time_index)
        supertable_df.supertable = pd.concat([supertable_df.supertable, cpt_procedures_columns], axis=1)
        logging.info(f"CPT procedures added to supertable.")

        # Step 20: Add gender code 
        supertable_df.supertable['gender_code'] = clinical_data.static_features.get("gender_code", 0)
        logging.info(f"Gender code added to supertable.")

        #Step 21: Add comorbidity data
        quan_deyo_ICD10_columns, quan_elix_ICD10_columns = self.comorbidity_staging(clinical_data.quan_deyo_ICD10, clinical_data.quan_elix_ICD10)
        clinical_data.quan_deyo_ICD10_staging = quan_deyo_ICD10_columns
        clinical_data.quan_elix_ICD10_staging = quan_elix_ICD10_columns
        logging.info(f"Comorbidity data added to Clinical Data.")

        logging.info(f"Supertable created with {len(supertable_df.supertable)} rows.")
        return supertable_df


###############################################################################
# DerivedFeatures
###############################################################################

class DerivedFeatures: 
    """Compute derived clinical features that are not directly measured."""

    def __init__(self, config: SepyDictConfig):
        self.config = config
        self.bounds = config.bounds
    
    def height_weight_postprocessing(
        self,
        super_table: pd.DataFrame,
        weight_col: str = "daily_weight_kg",
        height_col: str = "height_cm",
    ) -> None:
        """Fill missing height/weight using gender averages."""
        gender = super_table['gender'].iloc[0]
        df = super_table
 
        if df[weight_col].isnull().all():
            if gender == 'Male':
                df.iloc[0, df.columns.get_loc(weight_col)] = self.config.constants['default_weight_male']
                df.iloc[0, df.columns.get_loc(height_col)] = self.config.constants['default_height_male']
            elif gender == 'Female':
                df.iloc[0, df.columns.get_loc(weight_col)] = self.config.constants['default_weight_female']
                df.iloc[0, df.columns.get_loc(height_col)] = self.config.constants['default_height_female']
            else:
                df.iloc[0, df.columns.get_loc(weight_col)] = (self.config.constants['default_weight_male'] + self.config.constants['default_weight_female']) / 2
                df.iloc[0, df.columns.get_loc(height_col)] = (self.config.constants['default_height_male'] + self.config.constants['default_height_female']) / 2

        # Remove implausible values
        df[weight_col] = df[weight_col].where(
            (df[weight_col] >= self.config.constants['min_weight']) & (df[weight_col] <= self.config.constants['max_weight']),
            np.nan,
        )
        df[height_col] = df[height_col].where(df[height_col] > self.config.constants['min_height'], np.nan)

        first_valid_idx = df[height_col].first_valid_index()
        if first_valid_idx is not None:
            df[weight_col].loc[:first_valid_idx] = df[weight_col].loc[:first_valid_idx].bfill()
            df[height_col].loc[:first_valid_idx] = df[height_col].loc[:first_valid_idx].bfill()

        df[weight_col] = df[weight_col].ffill()
        df[height_col] = df[height_col].ffill()
        logging.info(f"Height and weight postprocessing completed.")

        return df


    def calculate_best_map(self, supertable: pd.DataFrame) -> pd.Series:
        """
        Vectorized calculation of best MAP for entire DataFrame.
        
        Args:
            df: DataFrame containing BP measurements
            
        Returns:
            Series with best MAP values
        """
        # Calculate MAP from arterial line
        map_line = np.where(
            (supertable['sbp_line'].notna() & supertable['dbp_line'].notna() & 
             ((supertable['sbp_line'] - supertable['dbp_line']) > 15)),
            (1/3) * supertable['sbp_line'] + (2/3) * supertable['dbp_line'],
            np.nan
        )
        
        # Calculate MAP from cuff (fallback)
        map_cuff = np.where(
            (supertable['sbp_cuff'].notna() & supertable['dbp_cuff'].notna() & 
             ((supertable['sbp_cuff'] - supertable['dbp_cuff']) > 15)),
            (1/3) * supertable['sbp_cuff'] + (2/3) * supertable['dbp_cuff'],
            np.nan
        )
        
        # Use arterial line if available, otherwise cuff
        best_map = np.where(pd.notna(map_line), map_line, map_cuff)
        
        # Validate physiological range
        upper_bound = float(self.bounds.loc[self.bounds["location in supertable"] == "best_map", "physical upper bound"].values[0])
        lower_bound = float(self.bounds.loc[self.bounds["location in supertable"] == "best_map", "physical lower bound"].values[0])
        best_map = np.where(
            (best_map >= np.array(lower_bound).repeat(len(best_map))) & (best_map <= np.array(upper_bound).repeat(len(best_map))),
            best_map,
            np.nan
        )
        
        df = pd.DataFrame(best_map, columns=['best_map'], index=supertable.index, dtype='float32')
        logging.info(f"Best MAP calculated.")
        return df

    
    def calculate_pulse_pressure(self, supertable: pd.DataFrame) -> pd.Series:
        """Calculation of pulse pressure from systolic and diastolic measurements."""
        pulse_pressure_line = np.where(
            (supertable['sbp_line'].notna() & supertable['dbp_line'].notna() & 
             ((supertable['sbp_line'] - supertable['dbp_line']) > 15)),
            supertable['sbp_line'] - supertable['dbp_line'],
            np.nan)
        pulse_pressure_cuff = np.where(
            (supertable['sbp_cuff'].notna() & supertable['dbp_cuff'].notna() & 
             ((supertable['sbp_cuff'] - supertable['dbp_cuff']) > 15)),
            supertable['sbp_cuff'] - supertable['dbp_cuff'],
            np.nan)
        pulse_pressure = np.where(pd.notna(pulse_pressure_line), pulse_pressure_line, pulse_pressure_cuff)
        df = pd.DataFrame(pulse_pressure, columns=['pulse_pressure'], index=supertable.index, dtype='float32')
        logging.info(f"Pulse pressure calculated.")
        return df

    def fio2_decimal(self, supertable: pd.DataFrame, fio2_column: str = 'fio2') -> None:
        """Convert FiO2 to decimal format if it's in percentage."""
        def fio2_row(row, fio2_column=fio2_column):
            if row[fio2_column] <= 1.0:
                return row[fio2_column]
            else:
                return row[fio2_column]/100
        
        supertable[fio2_column] = supertable.apply(fio2_row, axis=1)
        logging.info(f"FiO2 converted to decimal format.")
        return supertable

    def calculate_nl(self, supertable: pd.DataFrame, neutrophils_col: str = 'neutrophils', lymphocytes_col: str = 'lymphocyte') -> None:
        """Calculate neutrophil to lymphocyte ratio."""
        if neutrophils_col not in supertable.columns:
            raise ValueError(f"Columns {neutrophils_col} not found in supertable")
        if lymphocytes_col not in supertable.columns:
            raise ValueError(f"Columns {lymphocytes_col} not found in supertable")
        n_to_l = supertable[neutrophils_col]/supertable[lymphocytes_col]
        df = pd.DataFrame(n_to_l, columns=['n_to_l'], index=supertable.index, dtype='float32')
        logging.info(f"Neutrophil to lymphocyte ratio calculated.")
        return df

    def calculate_pf(self, supertable: pd.DataFrame, spo2_col: str = 'spo2', 
                pao2_col: str = 'partial_pressure_of_oxygen_(pao2)', 
                fio2_col: str = 'vent_fio2') -> None:
        """Calculate P:F ratios using SpO2 and PaO2."""
        if spo2_col not in supertable.columns:
            raise ValueError(f"Columns {spo2_col} not found in supertable")
        if pao2_col not in supertable.columns:
            raise ValueError(f"Columns {pao2_col} not found in supertable")
        if fio2_col not in supertable.columns:
            raise ValueError(f"Columns {fio2_col} not found in supertable")
        
        df = pd.DataFrame(index=supertable.index)
        df[f's2f_{fio2_col}'] = supertable[spo2_col]/supertable[fio2_col]
        df[f'p2f_{fio2_col}'] = supertable[pao2_col]/supertable[fio2_col]
        logging.info(f"P:F ratios calculated.")
        return df

    def _single_pressor_by_weight(self, row: pd.Series, single_pressors_name: str) -> float:
        """Calculate single vasopressor dose adjusted by weight."""
        if single_pressors_name == 'vasopressin':
            val = row[single_pressors_name]
        elif row[single_pressors_name + '_dose_unit'] == 'mcg/min':
            val = row[single_pressors_name]/row['daily_weight_kg']
        elif row[single_pressors_name + '_dose_unit'] == 'mcg/kg/min':
            val = row[single_pressors_name]
        else:
            val = row[single_pressors_name]
        return val

    def calculate_all_pressors(self, supertable: pd.DataFrame) -> None:
        """Calculate weight-adjusted doses for all vasopressors."""
        for val in self.config.vasopressor_names:
            supertable[val + '_dose_weight'] = supertable.apply(self._single_pressor_by_weight, single_pressors_name=val, axis=1)
        logging.info(f"Weight-adjusted doses for all vasopressors calculated.")
        return supertable

    def calculate_anion_gap(self, supertable: pd.DataFrame, sodium_col: str = 'sodium', 
                            chloride_col: str = 'chloride', 
                            bicarb_hco3_col: str = 'bicarb_(hco3)') -> None:
        """Calculate anion gap from electrolyte values."""
        if 'sodium' not in supertable.columns:
            raise ValueError(f"Columns {sodium_col} not found in supertable")
        if 'chloride' not in supertable.columns:
            raise ValueError(f"Columns {chloride_col} not found in supertable")
        if 'bicarb_(hco3)' not in supertable.columns:
            raise ValueError(f"Columns {bicarb_hco3_col} not found in supertable")
        anion_gap = supertable['sodium'] - (supertable['chloride'] + supertable['bicarb_(hco3)'])
        df = pd.DataFrame(anion_gap, columns=['anion_gap'], index=supertable.index, dtype='float32')
        logging.info(anion_gap.shape)
        logging.info(f"Anion gap calculated.")
        return df

    def calculate_worst_pf(self, supertable: pd.DataFrame, vent_status_col: str = 'vent_status', fio2_col: str = 'vent_fio2') -> None:
        """Calculate worst P:F ratios during ventilation."""
        if vent_status_col not in supertable.columns:
            raise ValueError(f"Columns {vent_status_col} not found in supertable")
        #select worse pf_pa when on vent
        worst_pf_pa = supertable[supertable[vent_status_col]>0][f'p2f_{fio2_col}'].min()
        if supertable[supertable[vent_status_col]>0][f'p2f_{fio2_col}'].size:
            worst_pf_pa_time = supertable[supertable[vent_status_col]>0][f'p2f_{fio2_col}'].idxmin(skipna=True)
        else: 
            worst_pf_pa_time = pd.NaT
        #select worse pf_sp when on vent
        worst_pf_sp = supertable[supertable[vent_status_col]>0][f's2f_{fio2_col}'].min() 
        if supertable[supertable[vent_status_col]>0][f's2f_{fio2_col}'].size:
            worst_pf_sp_time = supertable[supertable[vent_status_col]>0][f's2f_{fio2_col}'].idxmin(skipna=True)
        else: 
            worst_pf_sp_time = pd.NaT
        return worst_pf_pa, worst_pf_pa_time, worst_pf_sp, worst_pf_sp_time

    def flag_variables_pressors(self, supertable: pd.DataFrame) -> None:
        """Create indicator variables for vasopressor usage."""
        v_vasopressor_names_wo_dobutamine = self.config.vasopressor_names.copy()
        v_vasopressor_names_wo_dobutamine.remove('dobutamine')

        on_pressors = (supertable[v_vasopressor_names_wo_dobutamine].notna()).any(axis=1)
        on_dobutamine = (supertable['dobutamine'] > 0) 
        
        df = pd.DataFrame(index=supertable.index)
        df['on_pressors'] = on_pressors.astype('bool')
        df['on_dobutamine'] = on_dobutamine.astype('bool')
        logging.info(f"Indicator variables for vasopressor usage created.")
        return df

    def _create_elapsed_time(self, row: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp) -> float:
        """Calculate elapsed time between start and end for a given row timestamp."""
        if row - start > pd.Timedelta('0 days') and row - end <= pd.Timedelta('0 days'):
            return (row-start).days*24 + np.ceil((row-start).seconds/3600)
        elif row - start <= pd.Timedelta('0 days'):
            return 0
        elif row - end > pd.Timedelta('0 days'):
            return (end - start).days * 24 + np.ceil((end-start).seconds/3600)

    def create_elapsed_icu(self, supertable: pd.DataFrame, 
                           first_icu_start: pd.Timestamp, 
                           first_icu_end: pd.Timestamp) -> None:
        """Create elapsed ICU time column."""
        start = first_icu_start
        end = first_icu_end
        
        if start is None and end is None:
            elapsed_icu = [0]*len(supertable)
        elif start is None and end is not None:
            logging.ERROR(str(supertable.index[0]) + 'probably has an error in icu start and end times')
        elif start is not None and end is None:
            end = supertable.index[-1]
            elapsed_icu = pd.Series(supertable.index).apply(self._create_elapsed_time, start=start, end=end)
            elapsed_icu = elapsed_icu.values
        else:
            elapsed_icu = pd.Series(supertable.index).apply(self._create_elapsed_time, start=start, end=end)
            elapsed_icu = elapsed_icu.values
        df = pd.DataFrame(elapsed_icu, columns=['elapsed_icu'], index=supertable.index, dtype='float32')
        logging.info(f"Elapsed ICU time column created.")
        return df

    def create_elapsed_hosp(self, supertable: pd.DataFrame, 
                           first_hosp_start: pd.Timestamp, 
                           first_hosp_end: pd.Timestamp) -> None:
        """Create elapsed hospital time column."""
        start = first_hosp_start
        end = first_hosp_end
        
        elapsed_hosp = pd.Series(supertable.index)
        elapsed_hosp = elapsed_hosp.apply(self._create_elapsed_time, start=start, end=end)
        df = pd.DataFrame(elapsed_hosp.values, columns=['elapsed_hosp'], index=supertable.index, dtype='float32')
        logging.info(f"Elapsed hospital time column created.")
        return df
    

    def create_infection_sepsis_time(self, supertable: pd.DataFrame, 
                                    t_suspicion: pd.Timestamp, 
                                    t_sepsis3: pd.Timestamp) -> None:
        """Create infection and sepsis indicator columns based on time."""
        t_infection_idx = t_suspicion['t_suspicion'].first_valid_index()
        if t_infection_idx is not None:
            t_infection = t_suspicion.loc[t_infection_idx]['t_suspicion']
            infection = np.int32(supertable.index.values > t_infection)
        else:
            infection = [0]*len(supertable)
        
        col = [x for x in t_sepsis3.columns if 't_sepsis3' in x][0]
        t_sepsis3_idx = t_sepsis3[col].first_valid_index()
        if t_sepsis3_idx is not None:
            t_sepsis3 = t_sepsis3.loc[t_sepsis3_idx][col]
            sepsis = np.int32(supertable.index.values > t_sepsis3)
        else:
            sepsis = [0]*len(supertable)
        df_infection = pd.DataFrame(infection, columns=['infection'], index=supertable.index, dtype='int32')
        df_sepsis = pd.DataFrame(sepsis, columns=['sepsis'], index=supertable.index, dtype='int32')
        logging.info(f"Infection and sepsis indicator columns created.")
        return df_infection, df_sepsis