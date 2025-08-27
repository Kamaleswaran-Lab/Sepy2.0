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
import pickle
import warnings
import clinicalFeatures
import scoreCalculators
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd

from typing import Dict, Any, Tuple

# Import from proper modules to avoid duplication
from clinicalFeatures import ClinicalDataProcessor, DerivedFeatures, SepyDictConfig, ClinicalData, supertable

from scoreCalculators import (
    OrganSystemScoreCalculator, SIRSCalculator, SOFACalculator,
)




class sepyMaster:
    """
    Main class that manages the yearly data and configuration.
    Serves as a factory for creating sepyCSN instances.
    """
    def __init__(self, yearly_data_instance: Any, sepyDICTConfigs: Dict[str, Any], 
                 bounds: pd.DataFrame, save_dir: str):
        # Create configuration object
        self.config = SepyDictConfig.initialize_config(yearly_data_instance.v_quan_deyo_labels, 
                                                       yearly_data_instance.v_quan_elix_labels, 
                                                       bounds,
                                                       sepyDICTConfigs)
        self.bounds = bounds
        self.yearly_data_instance = yearly_data_instance
        self.save_dir = save_dir
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
                                    #   'clinical_notes': "csn", #TODO: Uncomment this later when we have clinical notes
                                      'radiology_notes': "csn",
                                      'icd_procedures': "csn",
                                      'cpt_procedures': "csn",
                                      'dialysis': "csn"}


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
        except Exception as e:
            logging.info("BEGIN LOG There were no %s data for identifier %s", name, identifier)
            print(e)
            empty_df = getattr(self.yearly_data_instance, df_name).iloc[0:0]
            empty_df.index.set_names(getattr(self.yearly_data_instance, df_name).index.names)
            return empty_df, filt_df_name
    
    def get_identifier(self, csn: Any, identifier_type: str) -> Any:
        """Get the identifier for a given CSN and identifier type."""
        if identifier_type == "csn":
            return csn
        elif (identifier_type == "pat_id") or (identifier_type == "patient_id"):
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
            save_dir=self.save_dir
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
        save_dir: str
    ):
        self.config = config
        self._data_processor = ClinicalDataProcessor(self.config)
        self._sofa_calculator = SOFACalculator(self.config)
        self._sirs_calculator = SIRSCalculator(self.config)
        self._organ_system_calculator = OrganSystemScoreCalculator(self.config)
        
        self._derived_features = DerivedFeatures(config)
        self.save_dir = save_dir
    
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
        self._super_table_created = False
        self._derived_features_calculated = False
        self._t_suspicion_calculated = False
        self._sofa_scores_calculated = False
        self._sirs_scores_calculated = False
        self._organ_system_scores_calculated = False
        self._sepsis_time_calculated = False

    def process(self) -> None:
        """
        Main processing pipeline that coordinates all the steps needed
        to analyze a patient encounter.
        """        
        logging.info(f"Processing CSN: {self.clinical_data.csn}")
        try:
            # 1. create supertables
            self.create_supertable()
            logging.info("Supertable created")

            # 2. Calculate derived features
            self._calculate_derived_features()
            logging.info("Derived features calculated")
            
            # 2. Calculate ICU stay 
            self.calculate_icu_stay()
            logging.info("ICU stay calculated")
            
            # 3. Calculate suspicion time
            self.calculate_t_susp()
            logging.info("T suspicion calculated")
            
            # 4. Calculate scores
            self.calculate_sirs_scores()
            self.calculate_sofa_scores()
            self.calculate_organ_system_scores()
            logging.info("Scores calculated")
            
            # 6. Create sepsis indicator label
            self.create_sepsis_flags()
            logging.info("Sepsis flags created")
            
            # 7. Calculate indicator variables
            self._calculate_indicator_variables()
            logging.info("Indicator variables calculated")
            
            # 7. Mark as processed (this is the final step)
            self._processed = True
            self.save_supertable()
            self.save_clinical_data()
        except Exception as e:
            logging.error(f"Error processing CSN: {self.clinical_data.csn}")
            logging.error(e)

    def save_supertable(self):
        self.supertable.supertable.to_pickle(f"{self.save_dir}/Supertables/{self.clinical_data.csn}.pkl")
        logging.info(f"Supertable saved to {self.save_dir}/Supertables/{self.clinical_data.csn}.pkl")
    
    def save_clinical_data(self):
        clinical_data_dict = {
            'static_features': self.clinical_data.static_features,
            'event_times': self.clinical_data.event_times,
            'flags': self.clinical_data.flags,
            'quan_deyo_ICD10': self.clinical_data.quan_deyo_ICD10,
            'quan_elix_ICD10': self.clinical_data.quan_elix_ICD10,
            'csn': self.clinical_data.csn,
            'pat_id': self.clinical_data.pat_id
        }
        with open(f"{self.save_dir}/ClinicalData/{self.clinical_data.csn}.pkl", "wb") as f:
            pickle.dump(clinical_data_dict, f)
        logging.info(f"Clinical data saved to {self.save_dir}/ClinicalData/{self.clinical_data.csn}.pkl")
    
    def create_supertable(self):
        self.supertable = self._data_processor.process_clinical_data(self.clinical_data)
        logging.info(f"Supertable created with {len(self.supertable.supertable)} rows.")
        self._super_table_created = True
    
    def calculate_icu_stay(self) -> None:
        """Calculate ICU stay start and end times"""
        if not self._super_table_created:
            raise ValueError("Must process clinical data into supertables before calculating ICU stay")
            
        icu_status = self.supertable.supertable['icu']
        if icu_status.sum() > 0:
            # mask all zeros (i.e. make nan) if there is a gap <=12hrs between ICU bed times then if fills it; otherwise it's zero
            gap_filled = ((icu_status.mask(icu_status == 0).fillna(method='ffill', limit=12)) + 
                          (icu_status.mask(icu_status == 0).fillna(method='bfill') * 0))
            self.gap_filled = gap_filled
            #converts index into a series 
            s = gap_filled.dropna().index.to_series()

            # if the delta between index vals is >1hr then mark it a start time
            start_time = s[s.diff(1) != pd.Timedelta('1 hours')].reset_index(drop=True)

            # if the reverse delta between index vals is > -1hr then mark it a end time
            end_time = s[s.diff(-1) != -pd.Timedelta('1 hours')].reset_index(drop=True)

            #makes a df with start, stop tuples
            times = pd.DataFrame({'start_time': start_time, 'end_time': end_time}, columns=['start_time', 'end_time'])
            
            self.clinical_data.add_event_time('first_icu_start', times.iloc[0]['start_time'])
            self.clinical_data.add_event_time('first_icu_end', times.iloc[0]['end_time'])
        
        else:
            self.clinical_data.add_event_time('first_icu_start', None)
            self.clinical_data.add_event_time('first_icu_end', None)
        
        logging.info("ICU stay calculated. First ICU start: %s, First ICU end: %s", self.clinical_data.event_times.get('first_icu_start'), self.clinical_data.event_times.get('first_icu_end'))

    def calculate_t_susp(self) -> None:
        """Calculate suspicion time"""
        abx_order_times = self.clinical_data.anti_infective_meds.med_order_time.unique()
        culture_times = self.clinical_data.cultures.order_time.unique()
        
        hours72 = pd.Timedelta(hours = 72)
        hours24 = pd.Timedelta(hours = 24)
        hours0 = pd.Timedelta(hours = 0)

        #t_susp if t_abx is first
        sus_abx_first = [(abx_t, clt_t) 
                   for abx_t in abx_order_times for clt_t in culture_times 
                   if (clt_t-abx_t) < hours24 and (clt_t-abx_t) > hours0]

        #t_susp if t_clt is first
        sus_clt_first = [(abx_t, clt_t)
                   for clt_t in culture_times for abx_t in abx_order_times
                   if (abx_t-clt_t) < hours72 and (abx_t-clt_t) > hours0]
        
        t_susp_list = sus_clt_first + sus_abx_first
        t_suspicion = pd.DataFrame(t_susp_list, columns=['t_abx','t_clt'])
        t_suspicion['t_suspicion'] = t_suspicion[['t_abx','t_clt']].min(axis=1)
        self.clinical_data.t_suspicion = t_suspicion.sort_values('t_suspicion')
        self._t_suspicion_calculated = True
        logging.info("T suspicion calculated") 

    def calculate_sirs_scores(self):
        """Calculate SIRS scores for Sepsis-2 criteria."""
        if not self._super_table_created:
            raise ValueError("Must process clinical data into supertables before calculating SIRS scores")
        
        # Use the SIRS calculator from scoreCalculators
        sirs_df = self._sirs_calculator.calculate_scores(self.supertable.supertable)
        self.clinical_data.sirs_scores = sirs_df.copy()
        sirs_df = sirs_df.rename(columns={'hourly_total':'sirs_total'})
        sirs_df = sirs_df.rename(columns={'delta_24h':'sirs_delta_24h'})
        self.supertable.supertable = pd.concat([self.supertable.supertable, sirs_df], axis = 1)
        
        self._sirs_scores_calculated = True
        logging.info("SIRS scores calculated")

    def calculate_sofa_scores(self):
        """Calculate SOFA scores for Sepsis-2 criteria."""
        if not self._super_table_created:
            raise ValueError("Must process clinical data into supertables before calculating SOFA scores")
        
        sofa_df = self._sofa_calculator.calculate_scores(self.supertable.supertable)
        self.clinical_data.sofa_scores = sofa_df.copy()
        sofa_df = sofa_df.rename(columns={'hourly_total':'sofa_total'})
        sofa_df = sofa_df.rename(columns={'delta_24h':'sofa_delta_24h'})
        sofa_df = sofa_df.rename(columns={'hourly_total_mod':'sofa_total_mod'})
        sofa_df = sofa_df.rename(columns={'delta_24h_mod':'sofa_delta_24h_mod'})
        self.supertable.supertable = pd.concat([self.supertable.supertable, sofa_df], axis = 1)
        self._sofa_scores_calculated = True
        logging.info("SOFA scores calculated")

    def calculate_organ_system_scores(self):
        """Calculate organ system scores for Sepsis-2 criteria."""
        if not self._super_table_created:
            raise ValueError("Must process clinical data into supertables before calculating organ system scores")
        
        organ_system_df = self._organ_system_calculator.calculate_scores(self.supertable.supertable)
        self.supertable.supertable = pd.concat([self.supertable.supertable, organ_system_df], axis = 1)
        self.clinical_data.organ_system_scores = organ_system_df
        self._organ_system_scores_calculated = True
        logging.info("Organ system scores calculated")

    def create_sepsis_flags(self):
        """
        Accepts- a SOFAPrep class instance
        Does- Runs all the prep and calc steps for SOFA score calculation
        Returns- A class instance with updated "super_table" and new "sofa_scores" data frame
        """
        if not self._super_table_created:
            raise ValueError("Must process clinical data into supertables before calculating sepsis time")
        
        sep3_time_df = self.calculate_sep3_time()
        sep3_time_df_mod = self.calculate_sep3_time_mod()

        # Set first sepsis 3 time in the flag dictionary
        df = sep3_time_df[sep3_time_df.notna().all(axis=1)].reset_index()
        if df.empty:
            logging.info("No sep3 times to add to flag dict")
            self.clinical_data.add_event_time('first_sep3_susp', None)
            self.clinical_data.add_event_time('first_sep3_SOFA', None)
            self.clinical_data.add_event_time('first_sep3_time', None)
        else:
            logging.info("adding first sep3 times to flag dict")
            self.clinical_data.add_event_time('first_sep3_susp', df['t_suspicion'][0])
            self.clinical_data.add_event_time('first_sep3_SOFA', df['t_SOFA'][0])
            self.clinical_data.add_event_time('first_sep3_time', df['t_sepsis3'][0])
            
        # Set first sepsis 3 time in the flag dictionary
        df = sep3_time_df_mod[sep3_time_df_mod.notna().all(axis=1)].reset_index()
        if df.empty:
            logging.info("No sep3_mod times to add to flag dict")
            self.clinical_data.add_event_time('first_sep3_susp_mod', None)
            self.clinical_data.add_event_time('first_sep3_SOFA_mod', None)
            self.clinical_data.add_event_time('first_sep3_time_mod', None)
        else:
            logging.info("adding first sep3_mod times to flag dict")
            self.clinical_data.add_event_time('first_sep3_susp_mod', df['t_suspicion'][0])
            self.clinical_data.add_event_time('first_sep3_SOFA_mod', df['t_SOFA_mod'][0])
            self.clinical_data.add_event_time('first_sep3_time_mod', df['t_sepsis3_mod'][0])
        
        self._sepsis_time_calculated = True
        logging.info("Sepsis flags created")

    def calculate_sep3_time(self, look_back=24, look_forward=12):
        """
        Calculates the Sepsis-3 time based on suspicion of infection and SOFA (Sequential Organ Failure Assessment) scores.
        Args:
        look_back (int, optional): The number of hours before suspicion time to look for SOFA events (default is 24).
        look_forward (int, optional): The number of hours after suspicion time to look for SOFA events (default is 12).
        """
        if not self._sofa_scores_calculated:
            raise ValueError("Must calculate SOFA scores before calculating sepsis time")
        
        if not self._t_suspicion_calculated:
            raise ValueError("Must calculate suspicion time before calculating sepsis time")
        
        # Initialize empty df to hold suspicion and sofa times
        sep3_time_df = pd.DataFrame(columns = ['t_suspicion','t_SOFA'])

        # get suspicion times from class
        suspicion_times = self.clinical_data.t_suspicion['t_suspicion'].sort_values().drop_duplicates()
        
        # if NO SUSPICION, then get all SOFA >2
        if suspicion_times.empty:
            df = self.clinical_data.sofa_scores
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
                potential_sofa_times = self.clinical_data.sofa_scores[self.clinical_data.sofa_scores['delta_24h'] >= 2]

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
        
        self.clinical_data.sep3_time = sep3_time_df
        logging.info("Sepsis-3 time calculated")
        return sep3_time_df

    def calculate_sep3_time_mod(self, look_back=24, look_forward=12):
        """
        Calculates the Sepsis-3 time based on suspicion of infection and SOFA (Sequential Organ Failure Assessment) scores.

        Args:
            look_back (int): The number of hours before suspicion time to look for SOFA events (default is 24).
            look_forward (int): The number of hours after suspicion time to look for SOFA events (default is 12).
        """
        # Initialize empty df to hold suspicion and sofa times
        sep3_time_df_mod = pd.DataFrame(columns = ['t_suspicion','t_SOFA_mod'])

        # get suspicion times from class
        suspicion_times = self.clinical_data.t_suspicion['t_suspicion'].sort_values().drop_duplicates()
        
        # if NO SUSPICION, then get  first SOFA >2
        if suspicion_times.empty:
            df = self.clinical_data.sofa_scores
            #get index of times when total change is >= 2
            if df.empty:
                pass
            else:
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

                potential_sofa_times_mod = self.clinical_data.sofa_scores[self.clinical_data.sofa_scores['delta_24h_mod'] >= 2].index.tolist()

                if not potential_sofa_times_mod:
                    sofa_times_mod.append(pd.to_datetime(float("NaN")))
                else:
                    sofa_times_mod.append(potential_sofa_times_mod[0])

        sep3_time_df_mod['t_suspicion'] = suspicion_times.tolist() 
        sep3_time_df_mod['t_SOFA_mod'] = sofa_times_mod
        sep3_time_df_mod['t_sepsis3_mod'] = sep3_time_df_mod.min(axis=1, skipna =False)
        
        all_sofa_times_mod = self.clinical_data.sofa_scores[self.clinical_data.sofa_scores['delta_24h_mod'] >= 2].reset_index()
        sep3_time_df_mod = all_sofa_times_mod['index'].to_frame().merge(sep3_time_df_mod, how='outer', left_on='index',right_on='t_SOFA_mod')        
        sep3_time_df_mod = sep3_time_df_mod.iloc[sep3_time_df_mod['index'].fillna(sep3_time_df_mod['t_suspicion']).argsort()].reset_index(drop=True).drop(columns=['t_SOFA_mod']).rename(columns={'index':'t_SOFA_mod'})
        
        self.clinical_data.sep3_time_mod = sep3_time_df_mod
        logging.info("Sepsis-3 time (modified) calculated")
        return sep3_time_df_mod
    
    def get_sepsis_status(self) -> Dict[str, Any]:
        """Get the final sepsis determination and related metrics"""
        if not self._sofa_scores_calculated or not self._sirs_scores_calculated or not self._sepsis_time_calculated:
            raise ValueError("Must calculate scores before getting sepsis status")
            
        return {
            'has_sepsis': self._determine_sepsis_status(),
            'sofa_max': self.clinical_data.sofa_scores.max(),
            'sirs_max': self.clinical_data.sirs_scores.max(),
            'infection_time': self.clinical_data.event_times.get('t_suspicion'),
            'sepsis_onset_time': self.clinical_data.event_times.get('t_sepsis3')
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


    def _determine_sepsis_status(self) -> bool:
        """Determine if the patient has sepsis based on the sepsis flags"""
        if not self._sepsis_time_calculated:
            raise ValueError("Must calculate sepsis time before determining sepsis status")
        
        return self.clinical_data.event_times.get('t_sepsis3') is not None

    def _calculate_derived_features(self) -> None:
        """Calculate all derived clinical features"""
        if not self._super_table_created:
            raise ValueError("Must process clinical data into supertables before calculating features")
            
        # process height weight
        self.supertable.supertable = self._derived_features.height_weight_postprocessing(self.supertable.supertable)
        
        # process MAP
        best_map_df = self._derived_features.calculate_best_map(self.supertable.supertable)
        self.supertable.supertable = pd.concat([self.supertable.supertable, best_map_df], axis=1)

        # process pulse pressure
        pulse_pressure_df = self._derived_features.calculate_pulse_pressure(self.supertable.supertable)
        self.supertable.supertable = pd.concat([self.supertable.supertable, pulse_pressure_df], axis=1)
        
        # process fio2
        self.supertable.supertable = self._derived_features.fio2_decimal(self.supertable.supertable, fio2_column='fio2')
        self.supertable.supertable = self._derived_features.fio2_decimal(self.supertable.supertable, fio2_column='vent_fio2')

        # process nl
        nl_df = self._derived_features.calculate_nl(self.supertable.supertable)
        self.supertable.supertable = pd.concat([self.supertable.supertable, nl_df], axis=1)

        # process pf
        pf_df = self._derived_features.calculate_pf(self.supertable.supertable)
        self.supertable.supertable = pd.concat([self.supertable.supertable, pf_df], axis=1)

        # process pressors
        self.supertable.supertable = self._derived_features.calculate_all_pressors(self.supertable.supertable)
        
        # anion gap 
        anion_gap_df = self._derived_features.calculate_anion_gap(self.supertable.supertable)
        self.supertable.supertable = pd.concat([self.supertable.supertable, anion_gap_df], axis = 1)
        
        #Worst pf
        worst_pf_pa, worst_pf_pa_time, worst_pf_sp, worst_pf_sp_time = self._derived_features.calculate_worst_pf(self.supertable.supertable)
        self.clinical_data.add_event_time('worst_pf_pa', worst_pf_pa)
        self.clinical_data.add_event_time('worst_pf_pa_time', worst_pf_pa_time)
        self.clinical_data.add_event_time('worst_pf_sp', worst_pf_sp)
        self.clinical_data.add_event_time('worst_pf_sp_time', worst_pf_sp_time)
        
        # flag pressors
        pressor_flag_df = self._derived_features.flag_variables_pressors(self.supertable.supertable)
        self.supertable.supertable = pd.concat([self.supertable.supertable, pressor_flag_df], axis=1)

    def _calculate_indicator_variables(self):
        # flag sepsis, t sus
        tsus_df, sepsis_df = self._derived_features.create_infection_sepsis_time(self.supertable.supertable, 
                                                                                 self.clinical_data.t_suspicion, 
                                                                                 self.clinical_data.sep3_time_mod)
        self.supertable.supertable = pd.concat([self.supertable.supertable, tsus_df, sepsis_df], axis=1)

        # create elapsed icu and hosp times
        elapsed_icu_df = self._derived_features.create_elapsed_icu(self.supertable.supertable, 
                                                                   self.clinical_data.event_times.get('first_icu_start'), 
                                                                   self.clinical_data.event_times.get('first_icu_end'))
        self.supertable.supertable = pd.concat([self.supertable.supertable, elapsed_icu_df], axis=1)
        elapsed_hosp_df = self._derived_features.create_elapsed_hosp(self.supertable.supertable, 
                                                                     self.clinical_data.event_times.get('hospital_admission_date_time'), 
                                                                     self.clinical_data.event_times.get('hospital_discharge_date_time'))
        self.supertable.supertable = pd.concat([self.supertable.supertable, elapsed_hosp_df], axis=1)

