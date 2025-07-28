# -*- coding: utf-8 -*-
"""
Kamaleswaran Labs
Author: Jack F. Regan
Edited: 2025-03-06
Version: 0.5
Changes:
  - cleaned up duplicated code by importing from proper modules
  - removed redundant classes and methods
  - streamlined imports and reduced file size
  - improved separation of concerns
"""
import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import time
import pandas as pd
import numpy as np

from comorbidipy import comorbidity
from typing import List, Dict, Any, Optional, Tuple

import utils
import sepyIMPORT

# Import from proper modules to avoid duplication
import clinicalFeatures
import importlib
importlib.reload(clinicalFeatures)
from clinicalFeatures import ClinicalDataProcessor, DerivedFeatures, EncounterDictionary
from scoreCalculators import (
     ScoreType, ScoreCalculatorFactory,
    DEFAULT_LOOKBACK_HOURS, DEFAULT_LOOKFORWARD_HOURS, SEPSIS_SCORE_THRESHOLD
)
from dataclasses import dataclass


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
    vasopressor_col_names: List[str]
    vent_col_names: List[str]
    vent_positive_vars: List[str]
    bp_cols: List[str]
    sofa_max_24h: List[str]
    fluids_med_names: List[str]
    fluids_med_names_generic: List[str]
    information_to_process: List[Dict[str, str]]
    lab_aggregation: Dict[str, str]
    dict_elements: List[Dict[str, Any]]
    write_dict_keys: List[str]
    constants: Dict[str, Any]
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        self.all_lab_col_names = self.numeric_lab_col_names + self.string_lab_col_names
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SepyDictConfig':
        """Create configuration instance from dictionary."""
        # Get field names from the dataclass
        field_names = {field.name for field in cls.__dataclass_fields__.values()}
        
        # Filter config_dict to only include known fields
        filtered_config = {k: v for k, v in config_dict.items() if k in field_names}
        
        return cls(**filtered_config)

class SepsisScoreCalculator:
    """Handles sepsis-related score calculations using the factory pattern."""
    
    def __init__(self):
        self.calculators = ScoreCalculatorFactory.create_all_calculators()
        self.sofa_calculator = self.calculators[ScoreType.SOFA]
        self.sirs_calculator = self.calculators[ScoreType.SIRS]
        self.qsofa_calculator = self.calculators[ScoreType.QSOFA]
    
    def calculate_sofa_coag_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """Vectorized coagulation SOFA calculation."""
        return self.sofa_calculator._calculate_coag_vectorized(df)
    
    def calculate_sofa_cardio_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """Vectorized cardiovascular SOFA calculation."""
        return self.sofa_calculator._calculate_cardio_vectorized(df)
    
    def calculate_sofa_resp_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """Vectorized respiratory SOFA calculation."""
        return self.sofa_calculator._calculate_resp_vectorized(df)
    
    def calculate_sofa_resp_sa_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """Vectorized respiratory SOFA calculation with SpO2/FiO2."""
        # This needs to be implemented in the SOFACalculator
        # For now, use the existing method
        return df.apply(lambda row: self.sofa_calculator._calculate_resp_single(row), axis=1)
    
    def calculate_sofa_renal_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """Vectorized renal SOFA calculation."""
        return self.sofa_calculator._calculate_renal_vectorized(df)
    
    def calculate_sofa_hep_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """Vectorized hepatic SOFA calculation."""
        return self.sofa_calculator._calculate_hep_vectorized(df)
    
    def calculate_sofa_neuro_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """Vectorized neurological SOFA calculation."""
        return self.sofa_calculator._calculate_neuro_vectorized(df)


class sepyDICT:
    """
    Main class that coordinates clinical data processing for sepsis evaluation.
    
    This class maintains the original interface while delegating responsibilities
    to specialized processor classes.

    Args:
        master_df: Master DataFrame containing all clinical data
        sepyDICTConfigs: Configuration dictionary for processing parameters
        bounds: DataFrame with threshold values and metadata for lab aggregation
    """
    def __init__(self, master_df: Any, sepyDICTConfigs: Dict[str, Any], bounds: pd.DataFrame):
        # Create configuration object
        self.config = SepyDictConfig.from_dict(sepyDICTConfigs)
        
        # Store original config and data for backward compatibility
        self.sepyDICTConfigs = sepyDICTConfigs
        self.bounds = bounds
        self.master_df = master_df

        # Initialize processor classes
        self.data_processor = ClinicalDataProcessor(self.config, bounds, master_df)
        self.score_calculator = SepsisScoreCalculator()
        self.dict_builder = EncounterDictionary(self.config)
        self.derived_features = DerivedFeatures(self.config)
        
        # Keep lab aggregation for backward compatibility
        self.labAGG = self.data_processor.labAGG
    
    def flag_dict(self):
        self.flags = {}
        
        #ID numbers
        self.flags['csn'] = self.csn
        self.flags['pt_id'] = self.pat_id
        
        #vent flags
        self.flags['y_vent_rows'] = 0
        self.flags['y_vent_start_time'] = 0
        self.flags['y_vent_end_time'] = 0
        self.flags['vent_start_time'] = pd.NaT
    
    def try_except(self, master_df: Any, identifier: Any, name: str) -> None:
        """Safely slice *master_df* by *identifier* and attach it to *instance*."""
        filt_df_name = name + "_PerCSN"
        df_name = "df_" + name

        try:
            source_df = getattr(master_df, df_name)
            if source_df.index.dtype == "O":
                setattr(self, filt_df_name, source_df.loc[[str(identifier)], :])
            else:
                setattr(self, filt_df_name, source_df.loc[[identifier], :])
            logging.info("The %s file was imported", name)
        except Exception: 
            empty_df = getattr(master_df, df_name).iloc[0:0]
            empty_df.index.set_names(getattr(master_df, df_name).index.names)
            setattr(self, filt_df_name, empty_df)
            logging.info("There were no %s data for identifier %s", name, identifier)
    
    
    def static_features_dict(self):

        #######################################
        # static_features: Patient demographic & encounter features that will not change during admisssion
        #######################################
        # from encounters file
        self.static_features = {}
        def safe_extract(df, column, default=None):
            try:
                return df.iloc[0, :][column] if not df.empty else default
            except (KeyError, IndexError):
                return default

        # Encounter features
        self.static_features['ed_arrival_source'] = safe_extract(self.encounters_PerCSN, 'ed_arrival_source')
        self.static_features['total_icu_days'] = safe_extract(self.encounters_PerCSN, 'total_icu_days', 0)
        self.static_features['discharge_status'] = safe_extract(self.encounters_PerCSN, 'discharge_status')
        self.static_features['discharge_to'] = safe_extract(self.encounters_PerCSN, 'discharge_to')
        self.static_features['encounter_type'] = safe_extract(self.encounters_PerCSN, 'encounter_type')
        self.static_features['age'] = safe_extract(self.encounters_PerCSN, 'age')
        self.static_features['admit_reason'] = safe_extract(self.encounters_PerCSN, 'admit_reason')

        # Demographics features
        self.static_features['gender'] = safe_extract(self.demographics_PerCSN, 'gender')
        self.static_features['gender_code'] = safe_extract(self.demographics_PerCSN, 'gender_code')
        self.static_features['race'] = safe_extract(self.demographics_PerCSN, 'race')
        self.static_features['race_code'] = safe_extract(self.demographics_PerCSN, 'race_code')
        self.static_features['ethnicity'] = safe_extract(self.demographics_PerCSN, 'ethnicity')
        self.static_features['ethnicity_code'] = safe_extract(self.demographics_PerCSN, 'ethnicity_code')
            
    def event_times_dict (self):
        #######################################
        # event_times: Key event times during a patients admission not otherwise specified
        #######################################
        def safe_extract(df, column, default=None):
            try:
                return df.iloc[0, :][column] if not df.empty else default
            except (KeyError, IndexError):
                return default

        self.event_times = {}    
        self.event_times ['ed_presentation_time'] = safe_extract(self.encounters_PerCSN, 'ed_presentation_time')
        self.event_times ['hospital_admission_date_time'] = safe_extract(self.encounters_PerCSN, 'hospital_admission_date_time')
        self.event_times ['hospital_discharge_date_time'] = safe_extract(self.encounters_PerCSN, 'hospital_discharge_date_time')
        self.event_times ['start_index'] = min(self.event_times['hospital_admission_date_time'], self.event_times['ed_presentation_time'])

        #Wait time
        self.flags['ed_wait_time'] = (self.event_times['hospital_admission_date_time'] - self.event_times['ed_presentation_time'])\
                                    .total_seconds() / 60
      
    def build_super_table_index(self):       
        """
        Builds the timestamp index for the super_table.
        """
        start_time = self.event_times['start_index']
        end_time = self.event_times['hospital_discharge_date_time']
        resample_frequency = self.config.constants['resample_frequency']
        self.super_table_time_index = pd.date_range(start_time, end_time, freq=resample_frequency)
        logging.info(f'SepyDICT- Super table index built with frequency {resample_frequency}')
    
    def make_dict_elements(self, imported):
        """
        Iterates over a set of predefined dictionary elements and executes corresponding methods 
        with optional arguments as specified in a configuration, logging each step if needed.
        """
        for step in self.config.dict_elements:
            method_name = step["method"]
            method = getattr(self, method_name)
            args = step.get("args", [])
            if args == "imported":
                method(imported)
            else:
                method(*args)

            if "log" in step:
                logging.info(step["log"])
     

    def create_supertable_pickles(self, csn: Any) -> None:
        """Main processing method that coordinates all data processing steps."""
        logging.info(f'SepyDICT- Creating sepyDICT instance for {csn}')
        filter_date_start_time = time.time()
        self.csn = csn

        # Set the patient ID based on the encounter
        try:
            self.pat_id = self.master_df.loc[csn,['pat_id']].iloc[0].item()
        except:
            self.pat_id = self.master_df.loc[csn,['pat_id']].iloc[0]
        
        # Get filtered DataFrames for each patient encounter
        for item in self.config.information_to_process:
            identifier = self.pat_id if item["id_type"] == "pat_id" else self.csn
            self.try_except(self.master_df, identifier, item["source"])
            
        logging.info('SepyDICT- Now making dictionary')
        self.make_dict_elements(self.master_df)
        logging.info('SepyDICT- Now calculating Sepsis-2')
        self.run_SEP2()
        logging.info('SepyDICT- Now calculating Sepsis-3')
        self.run_SEP3()
        self.derived_features.create_infection_sepsis_time(self)
        logging.info('SepyDICT- Now writing dictionary')
        self.write_dict()
        
        # Optimize memory usage of final DataFrames
        logging.info('SepyDICT- Optimizing memory usage')
        self.optimize_super_table_memory()
        
        # Log memory usage summary
        memory_summary = self.get_memory_usage_summary()
        logging.info(f'SepyDICT- Memory usage summary: {memory_summary}')
        
        logging.info(f'SepyDICT- Selecting data and writing this dict by CSN took {time.time() - filter_date_start_time}(s).')
    
    def calc_icu_stay(self):                
        if self.bed_status.icu.sum() > 0:
            # mask all zeros (i.e. make nan) if there is a gap <=12hrs between ICU bed times then if fills it; otherwise it's zero
            gap_filled = ((self.bed_status.mask(self.bed_status.icu == 0).icu.fillna(method='ffill', limit=12)) + 
                          (self.bed_status.mask(self.bed_status.icu == 0).icu.fillna(method='bfill') * 0))
            self.gap_filled = gap_filled
            #converts index into a series 
            s = gap_filled.dropna().index.to_series()

            # if the delta between index vals is >1hr then mark it a start time
            start_time = s[s.diff(1) != pd.Timedelta('1 hours')].reset_index(drop=True)

            # if the reverse delta between index vals is > -1hr then mark it a end time
            end_time = s[s.diff(-1) != -pd.Timedelta('1 hours')].reset_index(drop=True)

            #makes a df with start, stop tuples
            times = pd.DataFrame({'start_time': start_time, 'end_time': end_time}, columns=['start_time', 'end_time'])
            
            self.event_times ['first_icu_start'] = times.iloc[0]['start_time']

            self.event_times ['first_icu_end'] = times.iloc[0]['end_time']
        
           #self.event_times ['first_icu'] =  self.beds_PerCSN[self.beds_PerCSN.icu==1].sort_values('bed_location_start').bed_location_start.iloc[0]
        else:
           self.event_times ['first_icu_start'] = None
           self.event_times ['first_icu_end'] = None

    def calc_t_susp(self):
        self.abx_order_time = self.abx_staging.med_order_time.unique()
        self.culture_times = self.cultures_staging.order_time.unique()
        
        hours72 = pd.Timedelta(hours = 72)
        hours24 = pd.Timedelta(hours = 24)
        hours0 = pd.Timedelta(hours = 0)

        #t_susp if t_abx is first
        sus_abx_first = [(abx_t, clt_t) 
                   for abx_t in self.abx_order_time for clt_t in self.culture_times 
                   if (clt_t-abx_t) < hours24 and (clt_t-abx_t) > hours0]

        #t_susp if t_clt is first
        sus_clt_first = [(abx_t, clt_t)
                   for clt_t in self.culture_times for abx_t in self.abx_order_time
                   if (abx_t-clt_t) < hours72 and (abx_t-clt_t) > hours0]
        
        t_susp_list = sus_clt_first + sus_abx_first
        t_suspicion = pd.DataFrame(t_susp_list, columns=['t_abx','t_clt'])
        t_suspicion['t_suspicion'] = t_suspicion[['t_abx','t_clt']].min(axis=1)
        self.t_suspicion = t_suspicion.sort_values('t_suspicion')

    def fill_values(self, labs=None, vitals=None, gcs=None):
        """
        #NOTE : Not used!! 
            Accepts- Patient Dictionary and list of patient features to fill 
            Does- 1. Fwd fills each value for a max of 24hrs
              2. Back fills for a max of 24hrs from admission (i.e. for labs 1hr after admit)
        Returns- Patient Dictionary with filled patient features
        """
        if labs is None:
            v_all_lab_col_names = self.v_all_lab_col_names
        if vitals is None:
            v_vital_col_names = self.v_vital_col_names
        if gcs is None:
            v_gcs_col_names = self.v_gcs_col_names
            
        numerical_cols = v_all_lab_col_names + v_vital_col_names + v_gcs_col_names

        #Fwdfill to discharge    
        for col in numerical_cols:
            self.super_table[col] = self.super_table[col].ffill()
   
    def fill_pressor_values(self, v_vasopressor_names=None, v_vasopressor_units=None, v_vasopressor_dose=None):
        """
        Accepts- 1) Patient Dictionary
                    2) Lists of Initial vasopressor dose, vasopressor units, vasopressor weight based dose
           Does- Forward fills from first non-null value to the last non-null value. 
           Returns- 
           Notes- The assumption is that the last pressor is the last dose.
        """
        # Uses class variable for function
        if v_vasopressor_names is None:
            v_vasopressor_names = self.v_vasopressor_col_names
            
        if v_vasopressor_units is None:
            v_vasopressor_units= self.v_vasopressor_units
            
        if v_vasopressor_dose is None:
            v_vasopressor_dose = self.v_vasopressor_dose
            
        #create super_table variable
        df=self.super_table
        
        #fills the value for the initial vasopressor dose
        df[v_vasopressor_names]=df[v_vasopressor_names].apply(lambda columns: columns.loc[:columns.last_valid_index()].ffill())

        #fills the vasopressor name 
        df[v_vasopressor_units]=df[v_vasopressor_units].apply(lambda columns: columns.loc[:columns.last_valid_index()].ffill())
        
        #fills the weight based vasopressor dose
        df[v_vasopressor_dose]=df[v_vasopressor_dose].apply(lambda columns: columns.loc[:columns.last_valid_index()].ffill())

    def static_cci_to_supertable(self):
        #Get static features
        age = self.static_features['age']
        gender = self.static_features['gender']

        df = pd.DataFrame()
        df['code'] = self.diagnosis_PerCSN['dx_code_icd9'].values
        df['age'] = [age]*len(df)
        df['id'] = self.diagnosis_PerCSN.index

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
        df['code'] = self.diagnosis_PerCSN['dx_code_icd10'].values
        df['age'] = [age]*len(df)
        df['id'] = self.diagnosis_PerCSN.index

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

        self.super_table['age'] = [age]*len(self.super_table)
        self.super_table['gender'] = [gender]*len(self.super_table)
        self.super_table['cci9'] = [cci9]*len(self.super_table)
        self.super_table['cci10'] = [cci10]*len(self.super_table)

    

    def calc_all_SOFA(self, window: int = DEFAULT_LOOKBACK_HOURS) -> None:
        """
        Calculates the Sequential Organ Failure Assessment (SOFA) score for a patient based on various organ systems.
        
        Uses vectorized operations for improved performance and memory efficiency.
        
        Args:
            window: The rolling window size (in hours) used for calculating the delta of the SOFA score.
        """
        df = self.super_table
        
        # Use vectorized calculations where possible for better performance
        try:
            # Try vectorized approach first
            sofa_df = pd.DataFrame(index=df.index)
            
            # Use vectorized methods from score calculator
            sofa_df['SOFA_coag'] = self.score_calculator.calculate_sofa_coag_vectorized(df)
            sofa_df['SOFA_cardio'] = self.score_calculator.calculate_sofa_cardio_vectorized(df)
            sofa_df['SOFA_resp'] = self.score_calculator.calculate_sofa_resp_vectorized(df)
            sofa_df['SOFA_resp_sa'] = self.score_calculator.calculate_sofa_resp_sa_vectorized(df)
            sofa_df['SOFA_renal'] = self.score_calculator.calculate_sofa_renal_vectorized(df)
            sofa_df['SOFA_hep'] = self.score_calculator.calculate_sofa_hep_vectorized(df)
            sofa_df['SOFA_neuro'] = self.score_calculator.calculate_sofa_neuro_vectorized(df)
            
            # Modified cardio calculation still uses row-wise (more complex logic)
            sofa_df['SOFA_cardio_mod'] = df.apply(self.SOFA_cardio_mod, axis=1).astype('int8')
            
        except Exception as e:
            logging.warning(f"Vectorized SOFA calculation failed, falling back to row-wise: {e}")
            # Fallback to original row-wise calculation
            sofa_df = pd.DataFrame(index=df.index, columns=[
                'SOFA_coag', 'SOFA_renal', 'SOFA_hep', 'SOFA_neuro',
                'SOFA_cardio', 'SOFA_cardio_mod', 'SOFA_resp', 'SOFA_resp_sa'
            ])
            
            sofa_df['SOFA_coag'] = df.apply(self.SOFA_coag, axis=1)
            sofa_df['SOFA_renal'] = df.apply(self.SOFA_renal, axis=1)
            sofa_df['SOFA_hep'] = df.apply(self.SOFA_hep, axis=1)
            sofa_df['SOFA_neuro'] = df.apply(self.SOFA_neuro, axis=1)
            sofa_df['SOFA_cardio'] = df.apply(self.SOFA_cardio, axis=1)
            sofa_df['SOFA_cardio_mod'] = df.apply(self.SOFA_cardio_mod, axis=1)        
            sofa_df['SOFA_resp'] = df.apply(self.SOFA_resp, axis=1)
            sofa_df['SOFA_resp_sa'] = df.apply(self.SOFA_resp_sa, axis=1)
            
        # Calculate NORMAL hourly totals for each row
        sofa_df['hourly_total'] = sofa_df[[
                                   'SOFA_coag',
                                   'SOFA_renal',
                                   'SOFA_hep',
                                   'SOFA_neuro',
                                   'SOFA_cardio',
                               'SOFA_resp']].sum(axis=1)
        
        # Calculate POST 24hr delta in total SOFA Score
        sofa_df['delta_24h'] = sofa_df['hourly_total'].\
        rolling(window=window, min_periods=24).\
        apply(lambda x: x.max() - x.min() if x.idxmax().value> x.idxmin().value else 0 ).tolist()
 
        # Calculate FIRST 24h delta in total SOFA score
        sofa_df.update(sofa_df.loc[sofa_df.index[0:24],['hourly_total']].\
        rolling(window=window, min_periods=1).max().rename(columns={'hourly_total':'delta_24h'}))

        # Calculate MODIFIED hourly totals for each row
        sofa_df['hourly_total_mod'] = sofa_df[[
                               'SOFA_coag',
                               'SOFA_renal',
                               'SOFA_hep',
                               'SOFA_neuro',
                                'SOFA_cardio_mod',
                               'SOFA_resp_sa']].sum(axis=1)
        
        # Calculate POST 24hr delta in total SOFA Score
        sofa_df['delta_24h_mod'] = sofa_df['hourly_total_mod'].\
        rolling(window=window, min_periods=24).\
        apply(lambda x: x.max() - x.min() if x.idxmax().value> x.idxmin().value else 0 ).tolist()
 
        # Calculate FIRST 24h delta in total SOFA score
        sofa_df.update(sofa_df.loc[sofa_df.index[0:24],['hourly_total_mod']].\
        rolling(window=window, min_periods=1).max().rename(columns={'hourly_total_mod':'delta_24h_mod'}))                
        
        # Optimize memory usage of SOFA scores DataFrame
        self.sofa_scores = self.data_processor.optimize_dataframe_memory(sofa_df)

    def optimize_super_table_memory(self) -> None:
        """
        Optimize the memory usage of the super_table DataFrame.
        
        This method should be called after all processing is complete.
        """
        if hasattr(self, 'super_table') and self.super_table is not None:
            logging.info("Optimizing super_table memory usage...")
            original_memory = self.super_table.memory_usage(deep=True).sum() / 1024**2  # MB
            
            self.super_table = self.data_processor.optimize_dataframe_memory(self.super_table)
            
            optimized_memory = self.super_table.memory_usage(deep=True).sum() / 1024**2  # MB
            memory_saved = original_memory - optimized_memory
            
            logging.info(f"Memory optimization complete. Saved {memory_saved:.2f} MB "
                        f"({(memory_saved/original_memory)*100:.1f}% reduction)")

    def get_memory_usage_summary(self) -> Dict[str, float]:
        """
        Get memory usage summary for all major DataFrames.
        
        Returns:
            Dictionary with memory usage in MB for each DataFrame
        """
        memory_usage = {}
        
        if hasattr(self, 'super_table') and self.super_table is not None:
            memory_usage['super_table'] = self.super_table.memory_usage(deep=True).sum() / 1024**2
            
        if hasattr(self, 'sofa_scores') and self.sofa_scores is not None:
            memory_usage['sofa_scores'] = self.sofa_scores.memory_usage(deep=True).sum() / 1024**2
            
        if hasattr(self, 'sirs_scores') and self.sirs_scores is not None:
            memory_usage['sirs_scores'] = self.sirs_scores.memory_usage(deep=True).sum() / 1024**2
            
        memory_usage['total'] = sum(memory_usage.values())
        return memory_usage

    def run_SEP2(self):
        """Calculate SIRS scores for Sepsis-2 criteria."""
        df = self.super_table
        
        # Use the SIRS calculator from scoreCalculators
        sirs_df = self.score_calculator.sirs_calculator.calculate_scores(df)
        
        # Store SIRS scores
        self.sirs_scores = self.data_processor.optimize_dataframe_memory(sirs_df)

    def run_SEP3(self):
        """
        Accepts- a SOFAPrep class instance
        Does- Runs all the prep and calc steps for SOFA score calculation
        Returns- A class instance with updated "super_table" and new "sofa_scores" data frame
        """
        self.calc_all_SOFA()
        self.calc_sep3_time()
        self.calc_sep3_time_mod()

        # Set first sepsis 3 time in the flag dictionary
        df = self.sep3_time[self.sep3_time.notna().all(axis=1)].reset_index()
        if df.empty:
            logging.info("No sep3 times to add to flag dict")
            self.flags['first_sep3_susp'] = None
            self.flags['first_sep3_SOFA'] = None
            self.flags['first_sep3_time'] = None
        else:
            logging.info("adding first sep3 times to flag dict")
            self.flags['first_sep3_susp'] = df['t_suspicion'][0]
            self.flags['first_sep3_SOFA'] = df['t_SOFA'][0]
            self.flags['first_sep3_time'] = df['t_sepsis3'][0]
            
        # Set first sepsis 3 time in the flag dictionary
        df = self.sep3_time_mod[self.sep3_time_mod.notna().all(axis=1)].reset_index()
        if df.empty:
            logging.info("No sep3_mod times to add to flag dict")
            self.flags['first_sep3_susp_mod'] = None
            self.flags['first_sep3_SOFA_mod'] = None
            self.flags['first_sep3_time_mod'] = None
        else:
            logging.info("adding first sep3_mod times to flag dict")
            self.flags['first_sep3_susp_mod'] = df['t_suspicion'][0]
            self.flags['first_sep3_SOFA_mod'] = df['t_SOFA_mod'][0]
            self.flags['first_sep3_time_mod'] = df['t_sepsis3_mod'][0]

    def calc_sep3_time(self, look_back=24, look_forward=12):
        """
        Calculates the Sepsis-3 time based on suspicion of infection and SOFA (Sequential Organ Failure Assessment) scores.
        Args:
        look_back (int, optional): The number of hours before suspicion time to look for SOFA events (default is 24).
        look_forward (int, optional): The number of hours after suspicion time to look for SOFA events (default is 12).
        """
        # Initialize empty df to hold suspicion and sofa times
        sep3_time_df = pd.DataFrame(columns = ['t_suspicion','t_SOFA'])

        # get suspicion times from class
        suspicion_times = self.t_suspicion['t_suspicion'].sort_values().drop_duplicates()
        
        # if NO SUSPICION, then get all SOFA >2
        if suspicion_times.empty:
            df = self.sofa_scores
            #get index of times when total change is >= 2
            sofa_times = df[df['hourly_total'] >= 2].index

            if sofa_times.empty:
                pass
            else:
                sofa_times = sofa_times.tolist()[0]

        # If SUSPICION time is present    
        else:    
            sofa_times = []
            for suspicion_time in suspicion_times:
                #look back portion of window (i.e. 24hrs before Tsuspicion)
                start_window_time = suspicion_time - pd.Timedelta(hours = look_back)

                #look forward portion of window (i.e. 12hrs after Tsuspicion)
                end_window_time = suspicion_time + pd.Timedelta(hours = look_forward)
                
                # get all SOFA that had a 2pt change in last 24hrs (this is calculated in SOFA table)
                potential_sofa_times = self.sofa_scores[self.sofa_scores['delta_24h'] >= 2]

                # keep times that are with in a suspicion window
                potential_sofa_times = potential_sofa_times.loc[start_window_time:end_window_time].index.tolist()

                if not potential_sofa_times:
                    sofa_times.append(float("NaN"))
                else:
                    sofa_times.append(potential_sofa_times[0])
        
        #this adds Tsofa and Tsusp and picks the min; it's the most basic Tsep calculator
        sep3_time_df['t_suspicion'] = pd.to_datetime(suspicion_times.tolist())
        sep3_time_df['t_SOFA'] = pd.to_datetime(sofa_times)
        sep3_time_df['t_sepsis3'] = sep3_time_df.min(axis=1, skipna =False)
        
        self.sep3_time = sep3_time_df

    def calc_sep3_time_mod(self, look_back=24, look_forward=12):
        """
        Calculates the Sepsis-3 time based on suspicion of infection and SOFA (Sequential Organ Failure Assessment) scores.

        Args:
            look_back (int): The number of hours before suspicion time to look for SOFA events (default is 24).
            look_forward (int): The number of hours after suspicion time to look for SOFA events (default is 12).
        """
        # Initialize empty df to hold suspicion and sofa times
        sep3_time_df_mod = pd.DataFrame(columns = ['t_suspicion','t_SOFA_mod'])

        # get suspicion times from class
        suspicion_times = self.t_suspicion['t_suspicion'].sort_values().drop_duplicates()
        
        # if NO SUSPICION, then get  first SOFA >2
        if suspicion_times.empty:
            df = self.sofa_scores
            #get index of times when total change is >= 2
            sofa_times_mod = df[df['hourly_total_mod'] >= 2].index

            if sofa_times_mod.empty:
                pass
            else:
                sofa_times_mod = sofa_times_mod.tolist()[0]

        # If SUSPICION time is present    
        else:    
            sofa_times_mod = []
            for suspicion_time in suspicion_times:
                #look back portion of window (i.e. 24hrs before Tsuspicion)
                start_window_time = suspicion_time - pd.Timedelta(hours = look_back)

                #look forward portion of window (i.e. 12hrs after Tsuspicion)
                end_window_time = suspicion_time + pd.Timedelta(hours = look_forward)

                potential_sofa_times_mod = self.sofa_scores[self.sofa_scores['delta_24h_mod'] >= 2].index.tolist()

                if not potential_sofa_times_mod:
                    sofa_times_mod.append(pd.to_datetime(float("NaN")))
                else:
                    sofa_times_mod.append(potential_sofa_times_mod[0])

        sep3_time_df_mod['t_suspicion'] = suspicion_times.tolist() 
        sep3_time_df_mod['t_SOFA_mod'] = sofa_times_mod
        sep3_time_df_mod['t_sepsis3_mod'] = sep3_time_df_mod.min(axis=1, skipna =False)
        
        all_sofa_times_mod = self.sofa_scores[self.sofa_scores['delta_24h_mod'] >= 2].reset_index()
        sep3_time_df_mod = all_sofa_times_mod['index'].to_frame().merge(sep3_time_df_mod, how='outer', left_on='index',right_on='t_SOFA_mod')        
        sep3_time_df_mod = sep3_time_df_mod.iloc[sep3_time_df_mod['index'].fillna(sep3_time_df_mod['t_suspicion']).argsort()].reset_index(drop=True).drop(columns=['t_SOFA_mod']).rename(columns={'index':'t_SOFA_mod'})
        
        self.sep3_time_mod = sep3_time_df_mod

    # Score calculation methods - delegate to score calculator
    def SOFA_resp(self, row: pd.Series, pf_pa: str = 'pf_pa', pf_sp: str = 'pf_sp') -> float:
        return self.score_calculator.sofa_calculator._calculate_resp_single(row)
    
    def SOFA_resp_sa(self, row: pd.Series, pf_pa: str = 'pf_pa', pf_sp: str = 'pf_sp') -> float:
        return self.score_calculator.sofa_calculator._calculate_resp_single(row)

    def SOFA_cardio(self, row: pd.Series, 
                   dopamine_dose_weight: str = 'dopamine_dose_weight',
                   epinephrine_dose_weight: str = 'epinephrine_dose_weight',
                   norepinephrine_dose_weight: str = 'norepinephrine_dose_weight',
                   dobutamine_dose_weight: str = 'dobutamine_dose_weight') -> float:
        return self.score_calculator.sofa_calculator._calculate_cardio_single(row)

    def SOFA_coag(self, row: pd.Series) -> float:
        return self.score_calculator.sofa_calculator._calculate_coag_single(row)

    def SOFA_neuro(self, row: pd.Series) -> float:
        return self.score_calculator.sofa_calculator._calculate_neuro_single(row)

    def SOFA_hep(self, row: pd.Series) -> float:
        return self.score_calculator.sofa_calculator._calculate_hep_single(row)

    def SOFA_renal(self, row: pd.Series) -> float:
        return self.score_calculator.sofa_calculator._calculate_renal_single(row)
    
    def SOFA_cardio_mod(self, row: pd.Series,
                       dopamine_dose_weight: str = 'dopamine_dose_weight',
                       epinephrine_dose_weight: str = 'epinephrine_dose_weight',
                       norepinephrine_dose_weight: str = 'norepinephrine_dose_weight',
                       dobutamine_dose_weight: str = 'dobutamine_dose_weight') -> float:
        return self.score_calculator.sofa_calculator._calculate_cardio_single(row)

    # SIRS score calculation methods - delegate to score calculator
    def SIRS_resp(self, row: pd.Series, resp_rate: str = 'unassisted_resp_rate', paco2: str = 'partial_pressure_of_carbon_dioxide_(paco2)') -> int:
        return self.score_calculator.sirs_calculator._calculate_resp_single(row)

    def SIRS_cardio(self, row: pd.Series, hr: str = 'pulse') -> int:
        return self.score_calculator.sirs_calculator._calculate_cardio_single(row)
    
    def SIRS_temp(self, row: pd.Series, temp: str = 'temperature') -> int:
        return self.score_calculator.sirs_calculator._calculate_temp_single(row)
    
    def SIRS_wbc(self, row: pd.Series, wbc: str = 'white_blood_cell_count') -> int:
        return self.score_calculator.sirs_calculator._calculate_wbc_single(row)



