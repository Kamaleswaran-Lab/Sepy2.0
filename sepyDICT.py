# -*- coding: utf-8 -*-
"""
Kamaleswaran Labs
Author: Jack F. Regan
Edited: 2025-03-06
Version: 0.4
Changes:
  - improved documentation
  - implemented yaml configuration file
  - added configuration management with dataclasses
  - refactored into separate classes for better separation of concerns
  - added memory optimization with categorical data types
  - enhanced type hints and documentation
  - implemented vectorization and pandas optimization
  - defined constants at module level
"""
import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import time
import pandas as pd
import numpy as np

from functools import reduce
from comorbidipy import comorbidity
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
from enum import Enum

import utils
import sepyIMPORT

# Import constants and configuration from scoreCalculators to avoid duplication
from scoreCalculators import (
    SepyDictConfig, ScoreType, ScoreCalculatorFactory,
    RESAMPLE_FREQUENCY, DEFAULT_WEIGHT_MALE, DEFAULT_WEIGHT_FEMALE, 
    DEFAULT_HEIGHT_MALE, DEFAULT_HEIGHT_FEMALE, GENDER_MALE, GENDER_FEMALE,
    MAP_THRESHOLD, TEMPERATURE_HIGH_F, TEMPERATURE_LOW_F, HEART_RATE_THRESHOLD,
    RESP_RATE_THRESHOLD, WBC_HIGH_THRESHOLD, WBC_LOW_THRESHOLD, PACO2_THRESHOLD,
    SOFA_PLATELETS_THRESHOLDS, SOFA_BILIRUBIN_THRESHOLDS, SOFA_CREATININE_THRESHOLDS,
    SOFA_GCS_THRESHOLDS, SOFA_PF_THRESHOLDS, SOFA_PF_SP_THRESHOLDS,
    DOPAMINE_HIGH_THRESHOLD, DOPAMINE_MID_THRESHOLD, DOPAMINE_LOW_THRESHOLD,
    EPINEPHRINE_HIGH_THRESHOLD, EPINEPHRINE_LOW_THRESHOLD, NOREPINEPHRINE_HIGH_THRESHOLD,
    NOREPINEPHRINE_LOW_THRESHOLD, DOBUTAMINE_LOW_THRESHOLD, DEFAULT_LOOKBACK_HOURS,
    DEFAULT_LOOKFORWARD_HOURS, SEPSIS_SCORE_THRESHOLD, FILL_LIMIT_HOURS,
    VENT_FILL_LIMIT, MAX_WEIGHT, MIN_WEIGHT, MIN_HEIGHT, MIN_MAP, MAX_MAP
)

# Import clinical feature processing classes
from clinicalFeatures import ClinicalDataProcessor, DerivedFeatures




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
    try_except_calls: List[Dict[str, str]]
    lab_aggregation: Dict[str, str]
    dict_elements: List[Dict[str, Any]]
    write_dict_keys: List[str]
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        self.all_lab_col_names = self.numeric_lab_col_names + self.string_lab_col_names
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SepyDictConfig':
        """Create configuration instance from dictionary."""
        return cls(**config_dict)





class EncounterDictionary:
    """Handles final dictionary creation and serialization."""
    
    def __init__(self, config: SepyDictConfig):
        self.config = config
    
    def write_dict(self, instance: Any) -> None:
        """Create a dictionary of key attributes from the instance."""
        encounter_keys = self.config.write_dict_keys
        encounter_dict = {key: getattr(instance, key) for key in encounter_keys}
        instance.encounter_dict = encounter_dict





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
        
        # For backward compatibility, maintain direct attribute access
        self.v_vital_col_names = self.config.vital_col_names
        self.v_numeric_lab_col_names = self.config.numeric_lab_col_names
        self.v_string_lab_col_names = self.config.string_lab_col_names
        self.v_all_lab_col_names = self.config.all_lab_col_names
        self.v_gcs_col_names = self.config.gcs_col_names
        self.v_bed_info = self.config.bed_info
        self.v_vasopressor_names = self.config.vasopressor_names
        self.v_vasopressor_units = self.config.vasopressor_units
        self.v_vasopressor_dose = self.config.vasopressor_dose
        self.v_vasopressor_col_names = self.config.vasopressor_col_names
        self.v_vent_col_names = self.config.vent_col_names
        self.v_vent_positive_vars = self.config.vent_positive_vars
        self.v_bp_cols = self.config.bp_cols
        self.v_sofa_max_24h = self.config.sofa_max_24h
        self.v_fluids_med_names = self.config.fluids_med_names
        self.v_fluids_med_names_generic = self.config.fluids_med_names_generic

        # Store original config and data for backward compatibility
        self.sepyDICTConfigs = sepyDICTConfigs
        self.bounds = bounds
        self.master_df = master_df

        # Initialize processor classes
        self.data_processor = ClinicalDataProcessor(self.config, bounds, master_df)
        self.dict_builder = EncounterDictionary(self.config)
        self.derived_features = DerivedFeatures(self.config)
        
        # Initialize Factory Pattern score calculators
        self.factory_calculators = ScoreCalculatorFactory.create_all_calculators(self.config)
        self.sofa_calculator = self.factory_calculators[ScoreType.SOFA]
        self.sirs_calculator = self.factory_calculators[ScoreType.SIRS]
        self.qsofa_calculator = self.factory_calculators[ScoreType.QSOFA]
        
        # Keep lab aggregation for backward compatibility
        self.labAGG = self.data_processor.labAGG
     
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
        for item in self.config.try_except_calls:
            identifier = self.pat_id if item["id_type"] == "pat_id" else self.csn
            self.data_processor.try_except(self.master_df, identifier, item["section"], self)
            
        logging.info('SepyDICT- Now making dictionary')
        self.make_dict_elements(self.master_df)
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

    def try_except(self, master_df: Any, csn: Any, name: str) -> None:
        """Delegate to data processor for backward compatibility."""
        self.data_processor.try_except(master_df, csn, name, self)

    def bin_labs(self) -> None:
        """Delegate to data processor."""
        self.data_processor.bin_labs(self)

    def bin_vitals(self) -> None:
        """Delegate to data processor."""
        self.data_processor.bin_vitals(self)

    def bin_gcs(self) -> None:
        """Delegate to data processor."""
        self.data_processor.bin_gcs(self)

    def bin_vent(self) -> None:
        """Delegate to data processor."""
        self.data_processor.bin_vent(self)

    def bin_vasopressors(self) -> None:
        """Delegate to data processor."""
        self.data_processor.bin_vasopressors(self)

    def bin_fluids(self) -> None:
        """Delegate to data processor."""
        self.data_processor.bin_fluids(self)

    def write_dict(self) -> None:
        """Delegate to dictionary builder."""
        self.dict_builder.write_dict(self)

    # SOFA score calculation methods - delegate to score calculator
    def SOFA_resp(self, row: pd.Series, pf_pa: str = 'pf_pa', pf_sp: str = 'pf_sp') -> float:
        return self.sofa_calculator.SOFA_resp(row, pf_pa, pf_sp)
    
    def SOFA_resp_sa(self, row: pd.Series, pf_pa: str = 'pf_pa', pf_sp: str = 'pf_sp') -> float:
        return self.sofa_calculator.SOFA_resp_sa(row, pf_pa, pf_sp)

    def SOFA_cardio(self, row: pd.Series, 
                   dopamine_dose_weight: str = 'dopamine_dose_weight',
                   epinephrine_dose_weight: str = 'epinephrine_dose_weight',
                   norepinephrine_dose_weight: str = 'norepinephrine_dose_weight',
                   dobutamine_dose_weight: str = 'dobutamine_dose_weight') -> float:
        return self.sofa_calculator.SOFA_cardio(row, dopamine_dose_weight, epinephrine_dose_weight, norepinephrine_dose_weight, dobutamine_dose_weight)

    def SOFA_coag(self, row: pd.Series) -> float:
        return self.sofa_calculator.SOFA_coag(row)

    def SOFA_neuro(self, row: pd.Series) -> float:
        return self.sofa_calculator.SOFA_neuro(row)

    def SOFA_hep(self, row: pd.Series) -> float:
        return self.sofa_calculator.SOFA_hep(row)

    def SOFA_renal(self, row: pd.Series) -> float:
        return self.sofa_calculator.SOFA_renal(row)
    
    def SOFA_cardio_mod(self, row: pd.Series,
                       dopamine_dose_weight: str = 'dopamine_dose_weight',
                       epinephrine_dose_weight: str = 'epinephrine_dose_weight',
                       norepinephrine_dose_weight: str = 'norepinephrine_dose_weight',
                       dobutamine_dose_weight: str = 'dobutamine_dose_weight') -> float:
        return self.sofa_calculator.SOFA_cardio_mod(row, dopamine_dose_weight, epinephrine_dose_weight, norepinephrine_dose_weight, dobutamine_dose_weight)

    # SIRS score calculation methods - delegate to score calculator
    def SIRS_resp(self, row: pd.Series, resp_rate: str = 'unassisted_resp_rate', paco2: str = 'partial_pressure_of_carbon_dioxide_(paco2)') -> int:
        return self.sirs_calculator.SIRS_resp(row, resp_rate, paco2)

    def SIRS_cardio(self, row: pd.Series, hr: str = 'pulse') -> int:
        return self.sirs_calculator.SIRS_cardio(row, hr)
    
    def SIRS_temp(self, row: pd.Series, temp: str = 'temperature') -> int:
        return self.sirs_calculator.SIRS_temp(row, temp)
    
    def SIRS_wbc(self, row: pd.Series, wbc: str = 'white_blood_cell_count') -> int:
        return self.sirs_calculator.SIRS_wbc(row, wbc)

    def calc_icu_stay(self):
        """Calculate ICU stay times - needs bed_status implementation"""
        # TODO: Implement bed_status processing in ClinicalDataProcessor
        # For now, set default values
        self.event_times = getattr(self, 'event_times', {})
        self.event_times['first_icu_start'] = None
        self.event_times['first_icu_end'] = None      

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

    # Delegate derived feature calculations to DerivedFeatures class
    def fill_height_weight(self, weight_col: str = 'daily_weight_kg', height_col: str = 'height_cm') -> None:
        """Delegate to derived features processor."""
        self.derived_features.fill_height_weight(self, weight_col, height_col)
    
    def calc_best_map(self, row: pd.Series) -> float:
        """Delegate to derived features processor."""
        return self.derived_features.calc_best_map(row)
    
    def calc_pulse_pressure(self, row: pd.Series) -> float:
        """Delegate to derived features processor."""
        return self.derived_features.calc_pulse_pressure(row)
    
    def best_map(self, v_bp_cols: Optional[List[str]] = None) -> None:
        """Delegate to derived features processor."""
        self.derived_features.best_map(self, v_bp_cols)
    
    def pulse_pressure(self, v_bp_cols: Optional[List[str]] = None) -> None:
        """Delegate to derived features processor."""
        self.derived_features.pulse_pressure(self, v_bp_cols)
    
    def fio2_decimal(self, fio2: str = 'fio2') -> None:
        """Delegate to derived features processor."""
        self.derived_features.fio2_decimal(self, fio2)
    
    def calc_nl(self, neutrophils: str = 'neutrophils', lymphocytes: str = 'lymphocyte') -> None:
        """Delegate to derived features processor."""
        self.derived_features.calc_nl(self, neutrophils, lymphocytes)
    
    def calc_pf(self, spo2: str = 'spo2', pao2: str = 'partial_pressure_of_oxygen_(pao2)', fio2: str = 'fio2') -> None:
        """Delegate to derived features processor."""
        self.derived_features.calc_pf(self, spo2, pao2, fio2)
    
    def single_pressor_by_weight(self, row: pd.Series, single_pressors_name: str) -> float:
        """Delegate to derived features processor."""
        return self.derived_features.single_pressor_by_weight(row, single_pressors_name)
    
    def calc_all_pressors(self) -> None:
        """Delegate to derived features processor."""
        self.derived_features.calc_all_pressors(self)
###########################################################################
########################## Vasopressor Clean Up ############################
###########################################################################
    def fill_values(self, 
                    labs = None, 
                    vitals = None, 
                    gcs = None):
        """
        Accepts- Patient Dictionary and list of patient features to fill 
        Does- 1. Fwd fills each value for a max of 24hrs
              2. Back fills for a max of 24hrs from admission (i.e. for labs 1hr after admit)
        Returns- Patient Dictionary with filled patient features
        """
        if labs is None:
            v_all_lab_col_names =self.v_all_lab_col_names
        if vitals is None:
            v_vital_col_names = self.v_vital_col_names
        if gcs is None:
            v_gcs_col_names = self.v_gcs_col_names
            
        numerical_cols = v_all_lab_col_names + v_vital_col_names + v_gcs_col_names

        #Fwdfill to discharge    
        for col in numerical_cols:
            self.super_table[col] = self.super_table[col].ffill()
        #self.super_table[numerical_cols]=self.super_table[numerical_cols].ffill(limit=24)
        #self.super_table[numerical_cols]=self.super_table[numerical_cols].bfill(limit=24)
   
    def fill_pressor_values(self,
                            v_vasopressor_names = None,
                            v_vasopressor_units = None,
                            v_vasopressor_dose = None):

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

    def calc_comorbidities(self):
        # calculate CCI etc. return a df
        pass
    
    def calc_worst_pf(self) -> None:
        """Delegate to derived features processor."""
        self.derived_features.calc_worst_pf(self)

    def flag_variables_pressors(self) -> None:
        """Delegate to derived features processor."""
        self.derived_features.flag_variables_pressors(self)
        
    def create_elapsed_time(self, row: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp) -> float:
        """Delegate to derived features processor."""
        return self.derived_features.create_elapsed_time(row, start, end)
    
    def create_elapsed_icu(self) -> None:
        """Delegate to derived features processor."""
        self.derived_features.create_elapsed_icu(self)
    
    def create_elapsed_hosp(self) -> None:
        """Delegate to derived features processor."""
        self.derived_features.create_elapsed_hosp(self)
    
    def create_infection_sepsis_time(self) -> None:
        """Delegate to derived features processor."""
        self.derived_features.create_infection_sepsis_time(self)
            
    def create_on_vent(self) -> None:
        """Delegate to derived features processor."""
        self.derived_features.create_on_vent(self)
        
            
    def calculate_anion_gap(self) -> None:
        """Delegate to derived features processor."""
        self.derived_features.calculate_anion_gap(self)

    def static_cci_to_supertable(self):
        #Get static features
        age = self.static_features['age']
        gender = self.static_features['gender']
        # race = self.static_features['race']
        # ethnicity = self.static_features['ethnicity']

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
        # self.super_table['race'] = [race]*len(self.super_table)
        # self.super_table['ethnicity'] = [ethnicity]*len(self.super_table)

        self.super_table['cci9'] = [cci9]*len(self.super_table)
        self.super_table['cci10'] = [cci10]*len(self.super_table)
    def create_bed_unit(self) -> None:
        """Delegate to derived features processor."""
        self.derived_features.create_bed_unit(self)
        
    def on_dialysis(self) -> None:
        """Delegate to derived features processor."""
        self.derived_features.on_dialysis(self)
    def dialysis_history(self) -> None:
        """Delegate to derived features processor."""
        self.derived_features.dialysis_history(self)
        
    def create_fluids_columns(self) -> None:
        """Delegate to derived features processor."""
        self.derived_features.create_fluids_columns(self)
    def make_dict_elements(self, imported):
        """
        Iterates over a set of predefined dictionary elements and executes corresponding methods 
        with optional arguments as specified in a configuration, logging each step if needed.
        Args:
            imported (object): This argument is included but not used in the current method. 
                                It may be reserved for future use or passed in by the caller for external interactions.
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
    def write_dict(self):
        """
        Creates a dictionary of key attributes from the instance and stores it as an attribute.
        """
        encounter_keys = self.config.write_dict_keys
        encounter_dict = {key: getattr(self, key) for key in encounter_keys}
        #write to the instance
        self.encounter_dict = encounter_dict


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
            sofa_df['SOFA_coag'] = self.sofa_calculator._calculate_coag_vectorized(df)
            sofa_df['SOFA_cardio'] = self.sofa_calculator._calculate_cardio_vectorized(df)
            sofa_df['SOFA_resp'] = self.sofa_calculator._calculate_resp_vectorized(df)
            sofa_df['SOFA_resp_sa'] = self.sofa_calculator._calculate_resp_vectorized(df)  # Using same method for both
            sofa_df['SOFA_renal'] = self.sofa_calculator._calculate_renal_vectorized(df)
            sofa_df['SOFA_hep'] = self.sofa_calculator._calculate_hep_vectorized(df)
            sofa_df['SOFA_neuro'] = self.sofa_calculator._calculate_neuro_vectorized(df)
            
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
        ######## Normal Calcs                
        # Calculate NOMRAL hourly totals for each row
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

        ######## Modified Calcs                
        # Calculate NOMRAL hourly totals for each row
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
        

###########################################################################
################# Run all The Sepsis 3 steps ##############################
###########################################################################     

    def run_SEP3(self):
        """
        Accepts- a SOFAPrep class instance
        Does- Runs all the prep and calc steps for SOFA score calculation
        Returns- A class instance with updated "super_table" and new "sofa_scores" data frame
        """
        #start_sofa_calc = time.time()
        self.calc_all_SOFA()
        #self.hourly_max_SOFA ()
        self.calc_sep3_time()
        self.calc_sep3_time_mod()

        ####Set first sepsis 3 time in the flag dictionary
        #Select the first row that has 3x values
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
            
            self.calc_sep3_time_mod()

        
        #Set first sepsis 3 time in the flag dictionary
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
###########################################################################
############################# Calc Tsepsis-3 ##############################
###########################################################################     
    def calc_sep3_time(self,
                       look_back = 24,
                       look_forward = 12):
        """
        Calculates the Sepsis-3 time based on suspicion of infection and SOFA (Sequential Organ Failure Assessment) scores.
        Args:
        look_back (int, optional): The number of hours before suspicion time to look for SOFA events (default is 24).
        look_forward (int, optional): The number of hours after suspicion time to look for SOFA events (default is 12).
        """
        
        # Initialize empty list to hold SOFA times in loops below 
        #t_SOFA_list = []
        
        # Initialize empty df to hold suspicion and sofa times
        sep3_time_df = pd.DataFrame(columns = ['t_suspicion','t_SOFA'])

        # get suspicion times from class
        suspicion_times = self.t_suspicion['t_suspicion'].sort_values().drop_duplicates()
        
        #### if NO SUSPICION, then get all SOFA >2
        if suspicion_times.empty:
            df = self.sofa_scores
            #get index of times when total change is >= 2
            sofa_times = df[df['hourly_total'] >= 2].index

            if sofa_times.empty:
                pass
            
            else:
                sofa_times = sofa_times.tolist()[0]

        #### If SUSPICION time is present    
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
                #logging.info("These are potential SOFA Times: {}".format(potential_sofa_times))

                if not potential_sofa_times:
                    sofa_times.append(float("NaN"))
                    #logging.info ("A NaN was appended")
                else:
                    sofa_times.append(potential_sofa_times[0])
                    #logging.info("This SOFA Score was appended: {}".format(potential_sofa_times[0]))
        
        #this adds Tsofa and Tsusp and picks the min; it's the most basic Tsep calculator
        sep3_time_df['t_suspicion'] = pd.to_datetime(suspicion_times.tolist())
        sep3_time_df['t_SOFA'] = pd.to_datetime(sofa_times)
        sep3_time_df['t_sepsis3'] = sep3_time_df.min(axis=1, skipna =False)
        
        #This adds all the Tsofas that did not become part of a Tsepsis tuple; probably unecessary 
        #all_sofa_times = self.sofa_scores[self.sofa_scores['delta_24h'] >= 2].reset_index()
        #sep3_time_df = all_sofa_times['index'].to_frame().merge(sep3_time_df, how='outer', left_on='index',right_on='t_SOFA')        
        #sep3_time_df = sep3_time_df.iloc[sep3_time_df['index'].fillna(sep3_time_df['t_suspicion']).argsort()].reset_index(drop=True).drop(columns=['t_SOFA']).rename(columns={'index':'t_SOFA'})

        self.sep3_time = sep3_time_df
###########################################################################
############################# Calc Tsepsis-3 MOD  #########################
###########################################################################    
    def calc_sep3_time_mod(self,
                       look_back = 24,
                       look_forward = 12):
        """
        Calculates the Sepsis-3 time based on suspicion of infection and SOFA (Sequential Organ Failure Assessment) scores.

        Args:
            look_back (int): The number of hours before suspicion time to look for SOFA events (default is 24).
            look_forward (int): The number of hours after suspicion time to look for SOFA events (default is 12).
        """
        # Initialize empty list to hold SOFA times in loops below 
        #t_SOFA_list = []
        
        # Initialize empty df to hold suspicion and sofa times
        sep3_time_df_mod = pd.DataFrame(columns = ['t_suspicion','t_SOFA_mod'])

        # get suspicion times from class
        suspicion_times = self.t_suspicion['t_suspicion'].sort_values().drop_duplicates()
        
        #### if NO SUSPICION, then get  first SOFA >2
        if suspicion_times.empty:
            df = self.sofa_scores
            #get index of times when total change is >= 2
            sofa_times_mod = df[df['hourly_total_mod'] >= 2].index

            if sofa_times_mod.empty:
                pass
            
            else:
                sofa_times_mod = sofa_times_mod.tolist()[0]

        #### If SUSPICION time is present    
        else:    

            sofa_times_mod = []
            for suspicion_time in suspicion_times:
                #look back portion of window (i.e. 24hrs before Tsuspicion)
                start_window_time = suspicion_time - pd.Timedelta(hours = look_back)

                #look forward portion of window (i.e. 12hrs after Tsuspicion)
                end_window_time = suspicion_time + pd.Timedelta(hours = look_forward)

# =============================================================================
#                 #hourly SOFA score df windowed to relevant times
#                 df = self.sofa_scores.loc[start_window_time:end_window_time]
# 
#                 #Establish SOFA baseline for the windowget first SOFA score
#                 if start_window_time <= self.event_times['start_index']:
#                     baseline = 0
#                 else:
#                     baseline = df['hourly_total'][0]
# 
# =============================================================================
                potential_sofa_times_mod = self.sofa_scores[self.sofa_scores['delta_24h_mod'] >= 2].index.tolist()
                #logging.info("These are potential SOFA Times: {}".format(potential_sofa_times))

                if not potential_sofa_times_mod:
                    sofa_times_mod.append(pd.to_datetime(float("NaN")))
                    #logging.info("A NaN was appended")
                else:
                    sofa_times_mod.append(potential_sofa_times_mod[0])
                    #logging.info("This SOFA Score was appended: {}".format(potential_sofa_times[0]))

        sep3_time_df_mod['t_suspicion'] = suspicion_times.tolist() 
        sep3_time_df_mod['t_SOFA_mod'] = sofa_times_mod
        sep3_time_df_mod['t_sepsis3_mod'] = sep3_time_df_mod.min(axis=1, skipna =False)
        
        all_sofa_times_mod = self.sofa_scores[self.sofa_scores['delta_24h_mod'] >= 2].reset_index()
        sep3_time_df_mod = all_sofa_times_mod['index'].to_frame().merge(sep3_time_df_mod, how='outer', left_on='index',right_on='t_SOFA_mod')        
        sep3_time_df_mod = sep3_time_df_mod.iloc[sep3_time_df_mod['index'].fillna(sep3_time_df_mod['t_suspicion']).argsort()].reset_index(drop=True).drop(columns=['t_SOFA_mod']).rename(columns={'index':'t_SOFA_mod'})
        
        self.sep3_time_mod = sep3_time_df_mod

    

