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
from clinicalFeatures import ClinicalDataProcessor, DerivedFeatures, SepyDictConfig, ClinicalData, supertable
from scoreCalculators import (
     ScoreType, ScoreCalculatorFactory,
    DEFAULT_LOOKBACK_HOURS, DEFAULT_LOOKFORWARD_HOURS, SEPSIS_SCORE_THRESHOLD
)
from dataclasses import dataclass




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


class sepyMaster:
    """
    Main class that manages the yearly data and configuration.
    Serves as a factory for creating sepyCSN instances.
    """
    def __init__(self, yearly_data_instance: Any, sepyDICTConfigs: Dict[str, Any], bounds: pd.DataFrame):
        # Create configuration object
        self.config = SepyDictConfig.initialize_config(yearly_data_instance.v_quan_deyo_labels, yearly_data_instance.v_quan_elix_labels, sepyDICTConfigs)
        self.bounds = bounds
        self.yearly_data_instance = yearly_data_instance
        
        self.dataframes_id_mapping = {'beds': "csn",
                                      'demographics': "pat_id",
                                      'encounters': "csn",
                                      'gcs': "csn",
                                      'cultures': "csn",
                                      'procedures': "csn",
                                      'vent': "csn",
                                      'diagnosis': "csn",
                                      'labs': "csn",
                                      'vasopressor_meds': "csn",
                                      'anti_infective_meds': "csn",
                                      'vitals': "csn",
                                      'infusion_meds': "csn",
                                      'quan_deyo_ICD10': "csn",
                                      'quan_elix_ICD10': "csn",
                                      'in_out_fluids': "csn", #TODO Change this name later 
                                      'clinical_notes': "csn",
                                      'radiology_notes': "csn",
                                      'icd_procedures': "csn",
                                      'cpt_procedures': "csn"}

        # Initialize shared processors that can be reused
        self.data_processor = ClinicalDataProcessor(self.config, bounds)
        self.score_calculator = SepsisScoreCalculator()

    def slice_master_dataframes(self, identifier: Any, name: str) -> Tuple[pd.DataFrame, str]:
        """Safely slice yearly_data_instance by identifier."""
        filt_df_name = name + "_PerCSN"
        df_name = "df_" + name

        try:
            source_df = getattr(self.yearly_data_instance, df_name)
            if source_df.index.dtype == "O":
                return source_df.loc[[str(identifier)], :], filt_df_name
            else:
                return source_df.loc[[identifier], :], filt_df_name
        except Exception: 
            empty_df = getattr(self.yearly_data_instance, df_name).iloc[0:0]
            empty_df.index.set_names(getattr(self.yearly_data_instance, df_name).index.names)
            logging.info("There were no %s data for identifier %s", name, identifier)
            return empty_df, filt_df_name
    
    def get_identifier(self, csn: Any, identifier_type: str) -> Any:
        """Get the identifier for a given CSN and identifier type."""
        if identifier_type == "csn":
            return csn
        elif identifier_type == "pat_id":
            return self.yearly_data_instance.df_encounters.loc[csn,['pat_id']].iloc[0]
        else:
            raise ValueError(f"Invalid identifier type: {identifier_type}")

    def create_csn_instance(self, csn: Any) -> 'sepyCSN':
        """Factory method to create a new sepyCSN instance."""
        # Pre-slice all required dataframes for this CSN
        sliced_data = {}
        for df_name, identifier_type in self.dataframes_id_mapping.items():
            identifier = self.get_identifier(csn, identifier_type)
            df, df_name = self.slice_master_dataframes(identifier, df_name)
            sliced_data[df_name] = df
        
        logging.info(f"Slicing done for these dataframes: {sliced_data.keys()}")
        # Create and return new sepyCSN instance
        return sepyCSN(
            csn=csn,
            pat_id = self.get_identifier(csn, "pat_id"),
            config=self.config,
            sliced_data=sliced_data,
            data_processor=self.data_processor,
            score_calculator=self.score_calculator
        )
    

class sepyCSN:
    """
    Orchestrator class that coordinates the processing pipeline for a single patient encounter (CSN).
    Responsibilities:
    1. Coordinates the processing workflow
    2. Manages interactions between different processors
    3. Maintains the processing state
    4. Provides a high-level interface for the application
    """
    def __init__(
        self,
        csn: Any,
        pat_id: Any,
        config: SepyDictConfig,
        sliced_data: Dict[str, pd.DataFrame],
        data_processor: ClinicalDataProcessor,
        score_calculator: SepsisScoreCalculator
    ):
        self.config = config
        self._data_processor = data_processor
        self._score_calculator = score_calculator
        self._derived_features = DerivedFeatures(config)
        
        # Initialize raw data
        self.clinical_data = ClinicalData.from_sliced_data(
            csn=csn, 
            pat_id=pat_id,
            config=self.config,
            sliced_data=sliced_data
        )

        logging.info("Clinical data initialized. Supertable time index created. Flags initialized. Event times initialized. Static features initialized.")
        logging.info(f"Supertable starts at {self.clinical_data.super_table_time_index[0]} and ends at {self.clinical_data.super_table_time_index[-1]}, with {len(self.clinical_data.super_table_time_index)} rows.")
        
        # Processing state
        self._processed = False
        self._derived_features_calculated = False
        self._super_table_created = False
        self._sepsis_time_calculated = False
        self._sofa_scores_calculated = False
        self._sirs_scores_calculated = False
    
    def create_supertable(self):
        self.supertable = self._data_processor.process_clinical_data(self.clinical_data)
        logging.info(f"Supertable created with {len(self.supertable.supertable)} rows.")
        self._super_table_created = True

    def process(self) -> None:
        """
        Main processing pipeline that coordinates all the steps needed
        to analyze a patient encounter.
        """
        # 1. Process clinical data
        self.clinical_data = self._data_processor.process_data(
            self.clinical_data,
            self.clinical_data.super_table_time_index
        )

        # 2. Calculate derived features
        self._calculate_derived_features()

        # 3. Calculate sepsis scores
        self._calculate_sepsis_scores()

        # 4. Mark as processed
        self._processed = True

    def _calculate_derived_features(self) -> None:
        """Calculate all derived clinical features"""
        if not self._processed:
            raise ValueError("Must process clinical data before calculating features")
            
        # Calculate MAP features
        self.clinical_data = self._derived_features.calculate_map_features(
            self.clinical_data
        )
        
        # Calculate ventilator features
        self.clinical_data = self._derived_features.calculate_vent_features(
            self.clinical_data
        )
        
        # Other feature calculations...
    
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


    def _calculate_sepsis_scores(self) -> None:
        """Calculate all sepsis-related scores"""
        if not self._processed:
            raise ValueError("Must process clinical data before calculating scores")
            
        # Calculate SOFA scores
        sofa_scores = self._score_calculator.calculate_sofa_scores(
            self.clinical_data
        )
        
        # Calculate SIRS scores
        sirs_scores = self._score_calculator.calculate_sirs_scores(
            self.clinical_data
        )
        
        # Store scores
        self.clinical_data.sofa_scores = sofa_scores
        self.clinical_data.sirs_scores = sirs_scores
        
        self._scores_calculated = True

    def get_sepsis_status(self) -> Dict[str, Any]:
        """Get the final sepsis determination and related metrics"""
        if not self._scores_calculated:
            raise ValueError("Must calculate scores before getting sepsis status")
            
        return {
            'has_sepsis': self._determine_sepsis_status(),
            'sofa_max': self.clinical_data.sofa_scores.max(),
            'sirs_max': self.clinical_data.sirs_scores.max(),
            'infection_time': self.clinical_data.event_times.get('infection_time'),
            'sepsis_onset_time': self.clinical_data.event_times.get('sepsis_onset_time')
        }

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of all processed data and calculations"""
        return {
            'csn': self.clinical_data.csn,
            'pat_id': self.clinical_data.pat_id,
            'processed': self._processed,
            'scores_calculated': self._scores_calculated,
            'flags': self.clinical_data.flags,
            'event_times': self.clinical_data.event_times
        }

class sepyCSNold:
    """
    Class for processing individual CSN data.
    Created per-CSN by sepyMaster.
    """
    def __init__(
        self, 
        csn: Any,
        pat_id: Any,
        config: SepyDictConfig,
        sliced_data: Dict[str, pd.DataFrame],
        data_processor: ClinicalDataProcessor,
        score_calculator: SepsisScoreCalculator
    ):
        self.csn = csn
        self.pat_id = pat_id
        self.config = config
        self.data_processor = data_processor
        self.score_calculator = score_calculator
        
        # Attach sliced dataframes
        for df_name, df in sliced_data.items():
            setattr(self, df_name, df)
    
    
   
    def process_clinical_features(self):
        """
        Applies custom processing for each clinical feature type 
        with optional arguments as specified in a configuration, logging each step if needed.
        """


        

        """ for step in self.config.dict_elements:
            method_name = step["method"]
            method = getattr(self, method_name)
            args = step.get("args", [])
            if args == "imported":
                method(imported)
            else:
                method(*args)

            if "log" in step:
                 logging.info(step["log"])
        """

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



