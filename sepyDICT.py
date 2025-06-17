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


class ClinicalDataProcessor:
    """Handles data binning, cleaning, and aggregation operations with memory optimization."""
    
    def __init__(self, config: SepyDictConfig, bounds: pd.DataFrame, master_df: Any):
        self.config = config
        self.bounds = bounds
        self.master_df = master_df

        # Setup lab aggregation functions
        self.labAGG = self._setup_lab_aggregation()
        
        # Define categorical columns for memory optimization
        self.categorical_columns = {
            'bed_unit': 'category',
            'bed_type': 'category', 
            'icu_type': 'category',
            'gender_code': 'category',
            'vent_status': 'int8',
            'on_vent': 'int8',
            'on_pressors': 'bool',
            'on_dobutamine': 'bool',
            'on_dialysis': 'int8',
            'history_of_dialysis': 'int8',
            'infection': 'int8',
            'sepsis': 'int8'
        }
    
    def _setup_lab_aggregation(self) -> Dict[str, Any]:
        """Setup lab aggregation functions based on configuration and bounds."""
        labAGG = self.config.lab_aggregation.copy()
        for lab in labAGG.keys():
            if len(self.bounds.loc[self.bounds['Location in SuperTable'] == lab]) > 0:
                labAGG[lab] = utils.agg_fn_wrapper(lab, self.bounds)
        return labAGG

    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by converting to appropriate data types.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Memory-optimized DataFrame
        """
        df_optimized = df.copy()
        
        # Convert categorical columns
        for col, dtype in self.categorical_columns.items():
            if col in df_optimized.columns:
                if dtype == 'category':
                    df_optimized[col] = df_optimized[col].astype('category')
                elif dtype in ['int8', 'bool']:
                    df_optimized[col] = df_optimized[col].astype(dtype)
        
        # Optimize numeric columns
        numeric_cols = df_optimized.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in self.categorical_columns:
                # Check if can be converted to smaller int type
                if df_optimized[col].dtype in ['int64', 'int32']:
                    col_min = df_optimized[col].min()
                    col_max = df_optimized[col].max()
                    
                    if col_min >= -128 and col_max <= 127:
                        df_optimized[col] = df_optimized[col].astype('int8')
                    elif col_min >= -32768 and col_max <= 32767:
                        df_optimized[col] = df_optimized[col].astype('int16')
                    elif col_min >= -2147483648 and col_max <= 2147483647:
                        df_optimized[col] = df_optimized[col].astype('int32')
                
                # Convert float64 to float32 where possible
                elif df_optimized[col].dtype == 'float64':
                    df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
        return df_optimized

    def create_efficient_time_series(self, start_time: pd.Timestamp, end_time: pd.Timestamp, 
                                   freq: str = RESAMPLE_FREQUENCY) -> pd.DatetimeIndex:
        """
        Create memory-efficient time series index.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            freq: Frequency string for resampling
            
        Returns:
            DatetimeIndex for time series
        """
        return pd.date_range(start=start_time, end=end_time, freq=freq)
    
    def try_except(self, master_df: Any, identifier: Any, name: str, instance: Any) -> None:
        """
        Extract a subset of DataFrame for specific identifier and data type.
        
        Args:
            master_df: Master DataFrame containing all data
            identifier: Patient identifier (CSN or pat_id)
            name: Data type name (e.g., 'demographics', 'labs')
            instance: Instance to set the filtered data on
        """
        # Construct attribute names for filtered DataFrame and source DataFrame
        filt_df_name = name + "_PerCSN"
        df_name = "df_" + name
        
        try:
            # Handle potential string/numeric index mismatch for all dataframes
            source_df = getattr(master_df, df_name)
            if source_df.index.dtype == 'O':
                # Convert identifier to string for string-based index
                setattr(instance, filt_df_name, source_df.loc[[str(identifier)],:])
            else:
                # Use identifier as-is for numeric index
                setattr(instance, filt_df_name, source_df.loc[[identifier],:])
            logging.info(f'The {name} file was imported')
        except Exception as e: 
            # Create empty DataFrame with same structure when identifier not found
            empty_df = getattr(master_df, df_name).iloc[0:0]
            # Preserve original index names in empty DataFrame
            empty_df.index.set_names(getattr(master_df, df_name).index.names)
            # Set empty DataFrame on instance
            setattr(instance, filt_df_name, empty_df)
            logging.info(f"There were no {name} data for identifier {identifier}")

    def bin_labs(self, instance: sepyIMPORT.sepyIMPORT) -> None:
        """
        Resamples and aligns patient lab data to a unified hourly time index.
        
        Uses optimized pandas operations for better performance and memory efficiency.
        """
        df = instance.labs_PerCSN
        if df.empty:
            df.index = df.index.get_level_values('collection_time')
            instance.labs_staging = pd.DataFrame(index=instance.super_table_time_index, columns=df.columns)
        else:
            df = df.reset_index('collection_time')
            
            # Pre-allocate dictionary for better performance
            resampled_data = {}
            
            # Process all lab columns at once using vectorized operations
            for key, agg_func in self.labAGG.items():
                if key in df.columns:
                    # Use more efficient resampling with explicit parameters
                    resampled_col = (df[[key, 'collection_time']]
                                   .set_index('collection_time')
                                   .resample(RESAMPLE_FREQUENCY, origin=instance.event_times['start_index'])
                                   .apply(agg_func)
                                   .reindex(instance.super_table_time_index))
                    resampled_data[key] = resampled_col[key]
            
            # Create DataFrame from dictionary (more efficient than concatenation)
            instance.labs_staging = pd.DataFrame(resampled_data, index=instance.super_table_time_index)
            
            # Optimize memory usage
            instance.labs_staging = self.optimize_dataframe_memory(instance.labs_staging)

    def bin_vitals(self, instance: sepyIMPORT.sepyIMPORT) -> None:
        """
        Resamples and aligns patient vital data to a unified hourly time index.
        
        Uses optimized pandas operations for better performance.
        """
        df = instance.vitals_PerCSN 
       
        if df.empty:
            instance.vitals_staging = pd.DataFrame(index=instance.super_table_time_index, columns=df.columns)
        else:
            # Pre-allocate dictionary for better performance
            resampled_data = {}
            
            for key in self.config.vital_col_names:
                if key in df.columns:
                    # Determine aggregation function
                    if len(self.bounds.loc[self.bounds['Location in SuperTable'] == key]) > 0:
                        agg_fn = utils.agg_fn_wrapper(key, self.bounds)
                    else:
                        agg_fn = "mean"
                    
                    # Use more efficient resampling
                    resampled_col = (df[[key, 'recorded_time']]
                                   .set_index('recorded_time')
                                   .resample(RESAMPLE_FREQUENCY, origin=instance.event_times['start_index'])
                                   .apply(agg_fn)
                                   .reindex(instance.super_table_time_index))
                    resampled_data[key] = resampled_col[key]
            
            # Create DataFrame from dictionary (more efficient than concatenation)
            instance.vitals_staging = pd.DataFrame(resampled_data, index=instance.super_table_time_index)
            
            # Optimize memory usage
            instance.vitals_staging = self.optimize_dataframe_memory(instance.vitals_staging)

    def bin_gcs(self, instance: sepyIMPORT.sepyIMPORT) -> None:
        """Resamples and aligns patient gcs data to a unified hourly time index."""
        df = instance.gcs_PerCSN
 
        if df.empty:
            df = df.drop(columns=['recorded_time'])
            instance.gcs_staging = pd.DataFrame(index=instance.super_table_time_index, columns=df.columns)
        else:
            new = pd.DataFrame([])
            for key in self.config.gcs_col_names:
                if len(self.bounds.loc[self.bounds['Location in SuperTable'] == key]) > 0:
                    agg_fn = utils.agg_fn_wrapper_min(key, self.bounds)
                else:
                    agg_fn = "min"
                col1 = df[[key, 'recorded_time']].resample('60min', on="recorded_time", origin=instance.event_times['start_index']).apply(agg_fn)
                new = pd.concat((new, col1), axis=1)
            instance.gcs_staging = new.reindex(instance.super_table_time_index)

    def bin_vent(self, instance: sepyIMPORT.sepyIMPORT) -> None:
        """Resamples and aligns patient ventilator data to a unified hourly time index."""
        df = instance.vent_PerCSN

        if df.empty:
            df = pd.DataFrame(columns=['vent_status','fio2'], index=instance.super_table_time_index)
            instance.vent_status = df.vent_status
            instance.vent_fio2 = df.fio2
        else:
            vent_start = df[df.vent_start_time.notna()].vent_start_time.values
            vent_stop = df[df.vent_stop_time.notna()].vent_stop_time.values
            
            if vent_start.size == 0:
                instance.flags['y_vent_rows'] = 1
                df['vent_status'] = np.where(df[self.config.vent_positive_vars].notnull().any(axis=1), 1, 0)
                
                if df['vent_status'].sum() > 0:
                    instance.flags['vent_start_time'] = df[df['vent_status'] > 0].recorded_time.iloc[0:1]
                else:
                    vent_start = []
                    
             #If there is a vent start, but no stop; add 6hrs to start time  
            if len(vent_start) != 0 and len(vent_stop) == 0:
                #flag identifies the presence of vent rows, and start time
                check_mech_vent_vars = ['vent_tidal_rate_set', 'peep']
                df['vent_status'] = np.where(df[check_mech_vent_vars].notnull().any(axis=1),1,0)
                
                #check if there are any "real" vent rows; if so 
                if df['vent_status'].sum()>0:
                    vent_start  =  df[df['vent_status']>0].recorded_time.iloc[0:1]
                else:
                    vent_start = []
                    
             #If there is a vent start, but no stop; add 6hrs to start time  
            if len(vent_start) != 0 and len(vent_stop) == 0:
                #flag identifies the presence of vent rows, and start time
                check_mech_vent_vars = ['vent_tidal_rate_set', 'peep']
                df['vent_status'] = np.where(df[check_mech_vent_vars].notnull().any(axis=1),1,0)
                
                #check if there are any "real" vent rows; if so 
                if df['vent_status'].sum()>0:
                    vent_stop  =  df[df['vent_status']>0].recorded_time.iloc[-1:]
            
            # Import utils for agg function
            import utils
            agg_fn = utils.agg_fn_wrapper('fio2', instance.bounds)
            if len(vent_start) == 0: #No valid mechanical ventilation values
                # vent_status and fio2 will get joined to super table later
                vent_fio2 = df[['recorded_time','fio2']].resample('60min',
                                             on = 'recorded_time',
                                             origin = instance.event_times['start_index']).apply(agg_fn) \
                                             .reindex(instance.super_table_time_index)
                df_dummy = pd.DataFrame(columns=['vent_status'], index=instance.super_table_time_index)
                # vent_status and fio2 will get joined to super table later
                vent_status = df_dummy.vent_status.values
            else:
            
                index = pd.Index([])
                vent_tuples = zip(vent_start, vent_stop )
    
                for pair in set(vent_tuples):
                    if pair[0] < pair[1]:
                        index = index.append( pd.date_range(pair[0], pair[1], freq='H'))
                    else: #In case of a mistake in start and stop recording
                        index = index.append( pd.date_range(pair[1], pair[0], freq='H'))  
                
                vent_status = pd.DataFrame(data=([1.0]*len(index)), columns =['vent_status'], index=index)
                
                #sets column to 1 if vent was on    
                vent_status = vent_status.resample('60min',
                                                   origin = instance.event_times['start_index']).mean() \
                                                   .reindex(instance.super_table_time_index)
                            
                vent_fio2 = df[['recorded_time','fio2']].resample('60min',
                                             on = 'recorded_time',
                                             origin = instance.event_times['start_index']).apply(agg_fn) \
                                             .reindex(instance.super_table_time_index)
                
        instance.super_table['on_vent'] = vent_status
        instance.super_table['vent_fio2'] = vent_fio2

    def create_bed_unit(self, instance: Any) -> None:
        """Create bed unit and related columns."""
        bedDf = instance.beds_PerCSN
        bed_start = bedDf['bed_location_start'].values
        bed_end = bedDf['bed_location_end'].values
        bed_unit = bedDf['bed_unit'].values

        instance.super_table['bed_unit'] = [0]*len(instance.super_table)

        for i in range(len(bedDf)):
            start = bed_start[i]
            end = bed_end[i]
            unit = bed_unit[i]
            idx = np.bitwise_and(instance.super_table.index >= start ,  instance.super_table.index <= end)
            instance.super_table.loc[idx, 'bed_unit'] = unit
            
        def map_bed_unit(bed_code, bed_mapping, var_type):
            unit = bed_mapping.loc[bed_mapping['bed_unit'] == bed_code][var_type].values
            if len(unit) > 0:
                return unit[0]
            else:
                return float("nan")
        
        try:
            instance.super_table['bed_type'] = instance.super_table['bed_unit'].apply(map_bed_unit, args = [instance.bed_to_unit_mapping, 'unit_type'])
            instance.super_table['icu_type'] = instance.super_table['bed_unit'].apply(map_bed_unit, args = [instance.bed_to_unit_mapping, 'icu_type'])
            # instance.super_table['hospital'] = instance.super_table['bed_unit'].apply(map_bed_unit, args = [instance.bed_to_unit_mapping, 'hospital'])
        except:
            instance.super_table['bed_type'] = [float("nan")]*len(instance.super_table)
            instance.super_table['icu_type'] = [float("nan")]*len(instance.super_table)
            # instance.super_table['hospital'] = [float("nan")]*len(instance.super_table)

    def on_dialysis(self, instance: Any) -> None:
        """Create dialysis status column."""
        dd = instance.dialysis_year.loc[instance.dialysis_year['Encounter Encounter Number'] == instance.csn]
        instance.super_table['on_dialysis'] = [0]*len(instance.super_table)
        for time in dd['Service Timestamp']:
            time = pd.to_datetime(time)
            instance.super_table.loc[(instance.super_table.index - time > pd.Timedelta('0 seconds')), 'on_dialysis'] = 1


class EncounterDictionary:
    """Handles final dictionary creation and serialization."""
    
    def __init__(self, config: SepyDictConfig):
        self.config = config
    
    def write_dict(self, instance: Any) -> None:
        """Create a dictionary of key attributes from the instance."""
        encounter_keys = self.config.write_dict_keys
        encounter_dict = {key: getattr(instance, key) for key in encounter_keys}
        instance.encounter_dict = encounter_dict


class DerivedFeatures:
    """Handles calculation and creation of derived features and columns."""
    
    def __init__(self, config: SepyDictConfig):
        self.config = config
    
    def fill_height_weight(self, instance: Any, weight_col: str = 'daily_weight_kg', height_col: str = 'height_cm') -> None:
        """
        Fill missing height and weight values with defaults based on gender.
        
        Uses vectorized operations and predefined constants for better performance.
        
        Args:
            instance: Instance with super_table and static_features
            weight_col: Column name for weight data
            height_col: Column name for height data
        """
        df = instance.super_table
        gender = instance.static_features.get('gender_code', 0)

        # If there is no weight or height substitute in average weight by gender 
        if df[weight_col].isnull().all():
            if gender == GENDER_MALE:
                df.iloc[0, df.columns.get_loc(weight_col)] = DEFAULT_WEIGHT_MALE
                df.iloc[0, df.columns.get_loc(height_col)] = DEFAULT_HEIGHT_MALE
            elif gender == GENDER_FEMALE:
                df.iloc[0, df.columns.get_loc(weight_col)] = DEFAULT_WEIGHT_FEMALE
                df.iloc[0, df.columns.get_loc(height_col)] = DEFAULT_HEIGHT_FEMALE
            else:
                # Use average of male & female for undefined gender
                df.iloc[0, df.columns.get_loc(weight_col)] = (DEFAULT_WEIGHT_MALE + DEFAULT_WEIGHT_FEMALE) / 2
                df.iloc[0, df.columns.get_loc(height_col)] = (DEFAULT_HEIGHT_MALE + DEFAULT_HEIGHT_FEMALE) / 2
         
        # Check for non-sensical values using vectorized operations
        df[weight_col] = df[weight_col].where(
            (df[weight_col] >= MIN_WEIGHT) & (df[weight_col] <= MAX_WEIGHT), 
            np.nan
        )
        df[height_col] = df[height_col].where(df[height_col] > MIN_HEIGHT, np.nan)

        # Use more efficient pandas methods for filling
        first_valid_idx = df[height_col].first_valid_index()
        if first_valid_idx is not None:
            df[weight_col].loc[:first_valid_idx] = df[weight_col].loc[:first_valid_idx].bfill()
            df[height_col].loc[:first_valid_idx] = df[height_col].loc[:first_valid_idx].bfill()

        # Forward fill to discharge
        df[weight_col] = df[weight_col].ffill()
        df[height_col] = df[height_col].ffill()

    def calc_best_map(self, row: pd.Series) -> float:
        """
        Calculate the best mean arterial pressure from available measurements.
        
        Uses constants and improved logic for better maintainability.
        
        Args:
            row: Pandas Series containing BP measurements
            
        Returns:
            Best MAP value or NaN if unavailable/invalid
        """
        # Check arterial line measurements first (more accurate)
        if (pd.notna(row.get('sbp_line')) and pd.notna(row.get('dbp_line')) and 
            (row['sbp_line'] - row['dbp_line']) > 15):
            best_map = (1/3) * row['sbp_line'] + (2/3) * row['dbp_line']
        # Check cuff measurements as fallback
        elif (pd.notna(row.get('sbp_cuff')) and pd.notna(row.get('dbp_cuff')) and 
              (row['sbp_cuff'] - row['dbp_cuff']) > 15):
            best_map = (1/3) * row['sbp_cuff'] + (2/3) * row['dbp_cuff']
        else:
            return np.nan
        
        # Validate MAP is within reasonable physiological range
        if best_map < MIN_MAP or best_map > MAX_MAP:
            return np.nan
            
        return best_map

    def calculate_best_map_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """
        Vectorized calculation of best MAP for entire DataFrame.
        
        Args:
            df: DataFrame containing BP measurements
            
        Returns:
            Series with best MAP values
        """
        # Calculate MAP from arterial line
        map_line = np.where(
            (df['sbp_line'].notna() & df['dbp_line'].notna() & 
             ((df['sbp_line'] - df['dbp_line']) > 15)),
            (1/3) * df['sbp_line'] + (2/3) * df['dbp_line'],
            np.nan
        )
        
        # Calculate MAP from cuff (fallback)
        map_cuff = np.where(
            (df['sbp_cuff'].notna() & df['dbp_cuff'].notna() & 
             ((df['sbp_cuff'] - df['dbp_cuff']) > 15)),
            (1/3) * df['sbp_cuff'] + (2/3) * df['dbp_cuff'],
            np.nan
        )
        
        # Use arterial line if available, otherwise cuff
        best_map = np.where(pd.notna(map_line), map_line, map_cuff)
        
        # Validate physiological range
        best_map = np.where(
            (best_map >= MIN_MAP) & (best_map <= MAX_MAP),
            best_map,
            np.nan
        )
        
        return pd.Series(best_map, index=df.index, dtype='float32')

    def calc_pulse_pressure(self, row: pd.Series) -> float:
        """Calculate pulse pressure from systolic and diastolic measurements."""
        if row[['sbp_line','dbp_line']].notnull().all() and (row['sbp_line'] - row['dbp_line']) > 15:
            pulse_pressure = row['sbp_line'] - row['dbp_line']
        elif row[['sbp_cuff','dbp_cuff']].notnull().all() and (row['sbp_cuff'] - row['dbp_cuff']) > 15:
            pulse_pressure = row['sbp_cuff'] - row['dbp_cuff']
        else:
            pulse_pressure = float("NaN")
        return pulse_pressure

    def best_map(self, instance: Any, v_bp_cols: Optional[List[str]] = None) -> None:
        """Add best MAP column to super_table."""
        if v_bp_cols is None:
            v_bp_cols = ['sbp_line', 'dbp_line', 'map_line', 'sbp_cuff', 'dbp_cuff', 'map_cuff']
        instance.super_table['best_map'] = instance.super_table[v_bp_cols].apply(self.calc_best_map, axis=1)

    def pulse_pressure(self, instance: Any, v_bp_cols: Optional[List[str]] = None) -> None:
        """Add pulse pressure column to super_table."""
        if v_bp_cols is None:
            v_bp_cols = ['sbp_line', 'dbp_line', 'map_line', 'sbp_cuff', 'dbp_cuff', 'map_cuff']
        instance.super_table['pulse_pressure'] = instance.super_table[v_bp_cols].apply(self.calc_pulse_pressure, axis=1)

    def fio2_decimal(self, instance: Any, fio2: str = 'fio2') -> None:
        """Convert FiO2 to decimal format if it's in percentage."""
        def fio2_row(row, fio2=fio2):
            if row[fio2] <= 1.0:
                return row[fio2]
            else:
                return row[fio2]/100
        
        df = instance.super_table
        df[fio2] = df.apply(fio2_row, axis=1)

    def calc_nl(self, instance: Any, neutrophils: str = 'neutrophils', lymphocytes: str = 'lymphocyte') -> None:
        """Calculate neutrophil to lymphocyte ratio."""
        df = instance.super_table
        df['n_to_l'] = df[neutrophils]/df[lymphocytes]

    def calc_pf(self, instance: Any, spo2: str = 'spo2', pao2: str = 'partial_pressure_of_oxygen_(pao2)', fio2: str = 'fio2') -> None:
        """Calculate P:F ratios using SpO2 and PaO2."""
        df = instance.super_table
        df['pf_sp'] = df[spo2]/df[fio2]
        df['pf_pa'] = df[pao2]/df[fio2]

    def single_pressor_by_weight(self, row: pd.Series, single_pressors_name: str) -> float:
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

    def calc_all_pressors(self, instance: Any) -> None:
        """Calculate weight-adjusted doses for all vasopressors."""
        df = instance.super_table
        for val in self.config.vasopressor_names:
            df[val + '_dose_weight'] = df.apply(self.single_pressor_by_weight, single_pressors_name=val, axis=1)

    def calculate_anion_gap(self, instance: Any) -> None:
        """Calculate anion gap from electrolyte values."""
        instance.super_table['anion_gap'] = instance.super_table['sodium'] - (instance.super_table['chloride'] + instance.super_table['bicarb_(hco3)'])

    def calc_worst_pf(self, instance: Any) -> None:
        """Calculate worst P:F ratios during ventilation."""
        df = instance.super_table
        #select worse pf_pa when on vent
        instance.flags['worst_pf_pa'] = df[df['vent_status']>0]['pf_pa'].min()
        if df[df['vent_status']>0]['pf_pa'].size:
            instance.flags['worst_pf_pa_time'] = df[df['vent_status']>0]['pf_pa'].idxmin(skipna=True)
        else: 
            instance.flags['worst_pf_pa_time'] = pd.NaT
        #select worse pf_sp when on vent
        instance.flags['worst_pf_sp'] = df[df['vent_status']>0]['pf_sp'].min() 
        if df[df['vent_status']>0]['pf_sp'].size:
            instance.flags['worst_pf_sp_time'] = df[df['vent_status']>0]['pf_sp'].idxmin(skipna=True)
        else: 
            instance.flags['worst_pf_sp_time'] = pd.NaT

    def flag_variables_pressors(self, instance: Any) -> None:
        """Create indicator variables for vasopressor usage."""
        v_vasopressor_names_wo_dobutamine = self.config.vasopressor_names.copy()
        v_vasopressor_names_wo_dobutamine.remove('dobutamine')

        on_pressors = (instance.super_table[v_vasopressor_names_wo_dobutamine].notna()).any(axis=1)
        on_dobutamine = (instance.super_table['dobutamine'] > 0) 
        
        instance.super_table['on_pressors'] = on_pressors.astype('bool')
        instance.super_table['on_dobutamine'] = on_dobutamine.astype('bool')

    def create_elapsed_time(self, row: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp) -> float:
        """Calculate elapsed time between start and end for a given row timestamp."""
        if row - start > pd.Timedelta('0 days') and row - end <= pd.Timedelta('0 days'):
            return (row-start).days*24 + np.ceil((row-start).seconds/3600)
        elif row - start <= pd.Timedelta('0 days'):
            return 0
        elif row - end > pd.Timedelta('0 days'):
            return (end - start).days * 24 + np.ceil((end-start).seconds/3600)

    def create_elapsed_icu(self, instance: Any) -> None:
        """Create elapsed ICU time column."""
        start = instance.event_times['first_icu_start']
        end = instance.event_times['first_icu_end']
        
        if start is None and end is None:
            instance.super_table['elapsed_icu'] = [0]*len(instance.super_table)
        elif start is None and end is not None:
            logging.ERROR(str(instance.csn) + 'probably has an error in icu start and end times')
        elif start is not None and end is None:
            end = instance.super_table.index[-1]
            instance.super_table['elapsed_icu'] = instance.super_table.index
            instance.super_table['elapsed_icu'] = instance.super_table['elapsed_icu'].apply(self.create_elapsed_time, start=start, end=end)
        else:
            instance.super_table['elapsed_icu'] = instance.super_table.index
            instance.super_table['elapsed_icu'] = instance.super_table['elapsed_icu'].apply(self.create_elapsed_time, start=start, end=end)

    def create_elapsed_hosp(self, instance: Any) -> None:
        """Create elapsed hospital time column."""
        start = instance.super_table.index[0]
        end = instance.super_table.index[-1]
        
        instance.super_table['elapsed_hosp'] = instance.super_table.index
        instance.super_table['elapsed_hosp'] = instance.super_table['elapsed_hosp'].apply(self.create_elapsed_time, start=start, end=end)

    def create_infection_sepsis_time(self, instance: Any) -> None:
        """Create infection and sepsis indicator columns based on time."""
        times = instance.sep3_time
        
        t_infection_idx = times['t_suspicion'].first_valid_index()
        if t_infection_idx is not None:
            t_infection = times['t_suspicion'].loc[t_infection_idx]
            instance.super_table['infection'] = np.int32(instance.super_table.index > t_infection)
        else:
            instance.super_table['infection'] = [0]*len(instance.super_table)
        
        t_sepsis3_idx = times['t_sepsis3'].first_valid_index()
        if t_sepsis3_idx is not None:
            t_sepsis3 = times['t_sepsis3'].loc[t_sepsis3_idx]
            instance.super_table['sepsis'] = np.int32(instance.super_table.index > t_sepsis3)
        else:
            instance.super_table['sepsis'] = [0]*len(instance.super_table)

    def dialysis_history(self, instance: Any) -> None:
        """Create dialysis history indicator column."""
        dialysis_history = instance.diagnosis_PerCSN.loc[(instance.diagnosis_PerCSN.dx_code_icd9 == '585.6') | (instance.diagnosis_PerCSN.dx_code_icd10 == 'N18.6')]
        if len(dialysis_history) == 0:
            instance.super_table['history_of_dialysis'] = [0]*len(instance.super_table)
        else:
            instance.super_table['history_of_dialysis'] = [1]*len(instance.super_table)

    def create_fluids_columns(self, instance: Any) -> None:
        """Create fluid medication columns."""
        infusionDf = instance.infusion_meds_PerCSN
        
        for med in self.config.fluids_med_names:
            instance.super_table[med] = [0]*len(instance.super_table)
            instance.super_table[med + '_dose'] = [float("nan")]*len(instance.super_table)
            df = infusionDf.loc[infusionDf['med_name'] == med]
            for j in range(len(df)):
                row = df.iloc[j]
                med_start = row['med_start']
                med_dose = row['med_action_dose']
                instance.super_table.loc[(abs(instance.super_table.index - med_start) < pd.Timedelta('60 min')) & (instance.super_table.index - med_start > pd.Timedelta('0 seconds')), med] = 1
                instance.super_table.loc[(abs(instance.super_table.index - med_start) < pd.Timedelta('60 min')) & (instance.super_table.index - med_start > pd.Timedelta('0 seconds')), med + '_dose'] = med_dose
        
        for med in self.config.fluids_med_names_generic:
            instance.super_table[med] = [0]*len(instance.super_table)
            instance.super_table[med + '_dose'] = [float("nan")]*len(instance.super_table)
            df = infusionDf.loc[infusionDf['med_name_generic'] == med]
            for j in range(len(df)):
                row = df.iloc[j]
                med_start = row['med_start']
                med_dose = row['med_action_dose']
                instance.super_table.loc[(abs(instance.super_table.index - med_start) < pd.Timedelta('60 min')) & (instance.super_table.index - med_start > pd.Timedelta('0 seconds')), med] = 1
                instance.super_table.loc[(abs(instance.super_table.index - med_start) < pd.Timedelta('60 min')) & (instance.super_table.index - med_start > pd.Timedelta('0 seconds')), med + '_dose'] = med_dose

    def create_on_vent(self, instance: Any) -> None:
        """Create ventilator status columns."""
        df = instance.vent_PerCSN
        instance.super_table['on_vent_old'] = instance.vent_status
        instance.super_table['vent_fio2_old'] = instance.vent_fio2

        if df.empty:
            # No vent times were found so return empty table with 
            # all flags remain set at zero
            df = pd.DataFrame(columns=['vent_status','fio2'], index=instance.super_table_time_index)
            # vent_status and fio2 will get joined to super table later
            vent_status = df.vent_status.values
            vent_fio2 = df.fio2.values
             
        else:
            #check to see there is a start & stop time
            vent_start = df[df.vent_start_time.notna()].vent_start_time.values
            vent_stop =  df[df.vent_stop_time.notna()].vent_stop_time.values
            
            #If no vent start time then examin vent_plus rows
            if len(vent_start) == 0:
                # identify rows that are real vent vals (i.e. no fio2 alone)
                check_mech_vent_vars = ['vent_tidal_rate_set', 'peep']
                df['vent_status'] = np.where(df[check_mech_vent_vars].notnull().any(axis=1),1,0)
                
                #check if there are any "real" vent rows; if so 
                if df['vent_status'].sum()>0:
                    vent_start  =  df[df['vent_status']>0].recorded_time.iloc[0:1]
                else:
                    vent_start = []
                    
             #If there is a vent start, but no stop; add 6hrs to start time  
            if len(vent_start) != 0 and len(vent_stop) == 0:
                #flag identifies the presence of vent rows, and start time
                check_mech_vent_vars = ['vent_tidal_rate_set', 'peep']
                df['vent_status'] = np.where(df[check_mech_vent_vars].notnull().any(axis=1),1,0)
                
                #check if there are any "real" vent rows; if so 
                if df['vent_status'].sum()>0:
                    vent_stop  =  df[df['vent_status']>0].recorded_time.iloc[-1:]
            
            # Import utils for agg function
            import utils
            agg_fn = utils.agg_fn_wrapper('fio2', instance.bounds)
            if len(vent_start) == 0: #No valid mechanical ventilation values
                # vent_status and fio2 will get joined to super table later
                vent_fio2 = df[['recorded_time','fio2']].resample('60min',
                                             on = 'recorded_time',
                                             origin = instance.event_times['start_index']).apply(agg_fn) \
                                             .reindex(instance.super_table_time_index)
                df_dummy = pd.DataFrame(columns=['vent_status'], index=instance.super_table_time_index)
                # vent_status and fio2 will get joined to super table later
                vent_status = df_dummy.vent_status.values
            else:
            
                index = pd.Index([])
                vent_tuples = zip(vent_start, vent_stop )
    
                for pair in set(vent_tuples):
                    if pair[0] < pair[1]:
                        index = index.append( pd.date_range(pair[0], pair[1], freq='H'))
                    else: #In case of a mistake in start and stop recording
                        index = index.append( pd.date_range(pair[1], pair[0], freq='H'))  
                
                vent_status = pd.DataFrame(data=([1.0]*len(index)), columns =['vent_status'], index=index)
                
                #sets column to 1 if vent was on    
                vent_status = vent_status.resample('60min',
                                                   origin = instance.event_times['start_index']).mean() \
                                                   .reindex(instance.super_table_time_index)
                            
                vent_fio2 = df[['recorded_time','fio2']].resample('60min',
                                             on = 'recorded_time',
                                             origin = instance.event_times['start_index']).apply(agg_fn) \
                                             .reindex(instance.super_table_time_index)
                
        instance.super_table['on_vent'] = vent_status
        instance.super_table['vent_fio2'] = vent_fio2

    def create_bed_unit(self, instance: Any) -> None:
        """Create bed unit and related columns."""
        bedDf = instance.beds_PerCSN
        bed_start = bedDf['bed_location_start'].values
        bed_end = bedDf['bed_location_end'].values
        bed_unit = bedDf['bed_unit'].values

        instance.super_table['bed_unit'] = [0]*len(instance.super_table)

        for i in range(len(bedDf)):
            start = bed_start[i]
            end = bed_end[i]
            unit = bed_unit[i]
            idx = np.bitwise_and(instance.super_table.index >= start ,  instance.super_table.index <= end)
            instance.super_table.loc[idx, 'bed_unit'] = unit
            
        def map_bed_unit(bed_code, bed_mapping, var_type):
            unit = bed_mapping.loc[bed_mapping['bed_unit'] == bed_code][var_type].values
            if len(unit) > 0:
                return unit[0]
            else:
                return float("nan")
        
        try:
            instance.super_table['bed_type'] = instance.super_table['bed_unit'].apply(map_bed_unit, args = [instance.bed_to_unit_mapping, 'unit_type'])
            instance.super_table['icu_type'] = instance.super_table['bed_unit'].apply(map_bed_unit, args = [instance.bed_to_unit_mapping, 'icu_type'])
            # instance.super_table['hospital'] = instance.super_table['bed_unit'].apply(map_bed_unit, args = [instance.bed_to_unit_mapping, 'hospital'])
        except:
            instance.super_table['bed_type'] = [float("nan")]*len(instance.super_table)
            instance.super_table['icu_type'] = [float("nan")]*len(instance.super_table)
            # instance.super_table['hospital'] = [float("nan")]*len(instance.super_table)

    def on_dialysis(self, instance: Any) -> None:
        """Create dialysis status column."""
        dd = instance.dialysis_year.loc[instance.dialysis_year['Encounter Encounter Number'] == instance.csn]
        instance.super_table['on_dialysis'] = [0]*len(instance.super_table)
        for time in dd['Service Timestamp']:
            time = pd.to_datetime(time)
            instance.super_table.loc[(instance.super_table.index - time > pd.Timedelta('0 seconds')), 'on_dialysis'] = 1


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
        self.score_calculator = SepsisScoreCalculator(self.config)
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
        return self.score_calculator.SOFA_resp(row, pf_pa, pf_sp)
    
    def SOFA_resp_sa(self, row: pd.Series, pf_pa: str = 'pf_pa', pf_sp: str = 'pf_sp') -> float:
        return self.score_calculator.SOFA_resp_sa(row, pf_pa, pf_sp)

    def SOFA_cardio(self, row: pd.Series, 
                   dopamine_dose_weight: str = 'dopamine_dose_weight',
                   epinephrine_dose_weight: str = 'epinephrine_dose_weight',
                   norepinephrine_dose_weight: str = 'norepinephrine_dose_weight',
                   dobutamine_dose_weight: str = 'dobutamine_dose_weight') -> float:
        return self.score_calculator.SOFA_cardio(row, dopamine_dose_weight, epinephrine_dose_weight, norepinephrine_dose_weight, dobutamine_dose_weight)

    def SOFA_coag(self, row: pd.Series) -> float:
        return self.score_calculator.SOFA_coag(row)

    def SOFA_neuro(self, row: pd.Series) -> float:
        return self.score_calculator.SOFA_neuro(row)

    def SOFA_hep(self, row: pd.Series) -> float:
        return self.score_calculator.SOFA_hep(row)

    def SOFA_renal(self, row: pd.Series) -> float:
        return self.score_calculator.SOFA_renal(row)
    
    def SOFA_cardio_mod(self, row: pd.Series,
                       dopamine_dose_weight: str = 'dopamine_dose_weight',
                       epinephrine_dose_weight: str = 'epinephrine_dose_weight',
                       norepinephrine_dose_weight: str = 'norepinephrine_dose_weight',
                       dobutamine_dose_weight: str = 'dobutamine_dose_weight') -> float:
        return self.score_calculator.SOFA_cardio_mod(row, dopamine_dose_weight, epinephrine_dose_weight, norepinephrine_dose_weight, dobutamine_dose_weight)

    # SIRS score calculation methods - delegate to score calculator
    def SIRS_resp(self, row: pd.Series, resp_rate: str = 'unassisted_resp_rate', paco2: str = 'partial_pressure_of_carbon_dioxide_(paco2)') -> int:
        return self.score_calculator.SIRS_resp(row, resp_rate, paco2)

    def SIRS_cardio(self, row: pd.Series, hr: str = 'pulse') -> int:
        return self.score_calculator.SIRS_cardio(row, hr)
    
    def SIRS_temp(self, row: pd.Series, temp: str = 'temperature') -> int:
        return self.score_calculator.SIRS_temp(row, temp)
    
    def SIRS_wbc(self, row: pd.Series, wbc: str = 'white_blood_cell_count') -> int:
        return self.score_calculator.SIRS_wbc(row, wbc)

    def calc_icu_stay(self):
                
        if self.data_processor.bed_status.icu.sum() > 0:
            # mask all zeros (i.e. make nan) if there is a gap <=12hrs between ICU bed times then if fills it; otherwise it's zero
            gap_filled = ((self.data_processor.bed_status.mask(self.data_processor.bed_status.icu == 0).icu.fillna(method='ffill', limit=12)) + 
                          (self.data_processor.bed_status.mask(self.data_processor.bed_status.icu == 0).icu.fillna(method='bfill') * 0))
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

    

