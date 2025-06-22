# -*- coding: utf-8 -*-
"""
Clinical Features and Data Processing Classes for Sepsis Detection
Author: Jack F. Regan
Edited: 2025-03-06
Version: 0.4
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

# Import constants from scoreCalculators to avoid duplication
from scoreCalculators import (
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


class ClinicalDataProcessor:
    """Handles data binning, cleaning, and aggregation operations with memory optimization."""
    
    def __init__(self, config: 'SepyDictConfig', bounds: pd.DataFrame, master_df: Any):
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


class DerivedFeatures:
    """Handles calculation and creation of derived features and columns."""
    
    def __init__(self, config: 'SepyDictConfig'):
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