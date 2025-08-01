from enum import Enum
from abc import ABC, abstractmethod
from typing import List, Dict
import warnings
import pandas as pd
import numpy as np
from akiFlagger import AKIFlagger




# Threshold constants for clinical scores
MAP_THRESHOLD = 70.0
TEMPERATURE_HIGH_F = 100.4
TEMPERATURE_LOW_F = 95.8
HEART_RATE_THRESHOLD = 90.0
RESP_RATE_THRESHOLD = 20.0
WBC_HIGH_THRESHOLD = 12.0
WBC_LOW_THRESHOLD = 4.0
PACO2_THRESHOLD = 32.0

# SOFA score thresholds
SOFA_PLATELETS_THRESHOLDS = [150, 100, 50, 20]
SOFA_BILIRUBIN_THRESHOLDS = [1.2, 2.0, 6.0, 12.0]
SOFA_CREATININE_THRESHOLDS = [1.2, 2.0, 3.5, 5.0]
SOFA_GCS_THRESHOLDS = [15, 13, 10, 6]
SOFA_PF_THRESHOLDS = [400, 300, 200, 100]
SOFA_PF_SP_THRESHOLDS = [302, 221, 142, 67]

# Vasopressor dose thresholds
DOPAMINE_HIGH_THRESHOLD = 15.0
DOPAMINE_MID_THRESHOLD = 5.0
DOPAMINE_LOW_THRESHOLD = 0.0
EPINEPHRINE_HIGH_THRESHOLD = 0.1
EPINEPHRINE_LOW_THRESHOLD = 0.0
NOREPINEPHRINE_HIGH_THRESHOLD = 0.1
NOREPINEPHRINE_LOW_THRESHOLD = 0.0
DOBUTAMINE_LOW_THRESHOLD = 0.0

# Time window constants
DEFAULT_LOOKBACK_HOURS = 24
DEFAULT_LOOKFORWARD_HOURS = 12
SEPSIS_SCORE_THRESHOLD = 2
FILL_LIMIT_HOURS = 24
VENT_FILL_LIMIT = 6


class ScoreType(Enum):
    """Enumeration of available score types."""
    SOFA = "sofa"
    SIRS = "sirs"
    APACHE = "apache"
    QSOFA = "qsofa"


class ScoreCalculatorBase(ABC):
    """Abstract base class for all score calculators."""
    
    @abstractmethod
    def calculate_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate scores for the entire DataFrame."""
        pass
    
    @abstractmethod
    def calculate_single_score(self, row: pd.Series) -> pd.DataFrame:
        """Calculate score for a single row."""
        pass
    
    @abstractmethod
    def get_score_components(self) -> List[str]:
        """Return list of score component names."""
        pass


class SOFACalculator(ScoreCalculatorBase):
    """SOFA Score Calculator implementation."""
    
    def __init__(self, config: dict):
        self.config = config
        self.components = [
            'SOFA_resp', 'SOFA_cardio', 'SOFA_coag', 
            'SOFA_neuro', 'SOFA_hep', 'SOFA_renal'
        ]
    
    def get_score_components(self) -> List[str]:
        """Return list of score component names."""
        return self.components.copy()
    
    def SOFA_resp(self,
                  row,
                  column_name: str):
        """
        Accepts- class instance, one row from "super_table", "pf" cols
        Does- Calculates Respiratory SOFA score
        Returns- Single value of Respiratory SOFA score
        """
        if row[column_name] < 100:
            val = 4
        elif row[column_name] < 200:
            val = 3
        elif row[column_name] < 300:
            val = 2
        elif row[column_name] < 400:
            val = 1
        elif row[column_name] >= 400:
            val = 0
        else: 
            val = float("NaN")
        return val
    
    def SOFA_cardio(self,
                    row,
                    dopamine_dose_weight ='dopamine_dose_weight',
                    epinephrine_dose_weight ='epinephrine_dose_weight',
                    norepinephrine_dose_weight  = 'norepinephrine_dose_weight',
                    dobutamine_dose_weight ='dobutamine_dose_weight'):
        """
        Accepts- class instance, one row from "super_table", weight based pressor cols
        Does- Calculates Cardio SOFA score
        Returns- Single value of Cardio SOFA score 
        """
        
        if ((row[dopamine_dose_weight] > 15) |
            (row[epinephrine_dose_weight] > 0.1) | 
            (row[norepinephrine_dose_weight] > 0.1)):
            val = 4
        elif ((row[dopamine_dose_weight] > 5) |
              ((row[epinephrine_dose_weight] > 0.0) & (row[epinephrine_dose_weight] <= 0.1)) | 
              ((row[norepinephrine_dose_weight] > 0.0) & (row[norepinephrine_dose_weight] <= 0.1))):
            val = 3
        elif (((row[dopamine_dose_weight] > 0.0) & (row[dopamine_dose_weight] <= 5))|
              (row[dobutamine_dose_weight] > 0)):
                val = 2
        elif (row['best_map'] < 70):
            val = 1
            
        elif (row['best_map'] >= 70):
            val = 0
        else:
            val = float("NaN")
        return val
    
    def SOFA_cardio_mod(self,
                    row,
                    dopamine_dose_weight ='dopamine_dose_weight',
                    epinephrine_dose_weight ='epinephrine_dose_weight',
                    norepinephrine_dose_weight  = 'norepinephrine_dose_weight',
                    dobutamine_dose_weight ='dobutamine_dose_weight'):
        """
        Accepts- class instance, one row from "super_table", weight based pressor cols
        Does- Calculates Cardio SOFA score
        Returns- Single value of Cardio SOFA score 
        """
        
        ## Mehak changed this code (second check on norepinephrine_dose_weight) on 8/1/2025 after checking with Dr. H
        ## CJ's code was:
        # if ((row[epinephrine_dose_weight] > 0.0) | (row[epinephrine_dose_weight] > 0.0)):
        #     val = 3
        # elif ((row[dopamine_dose_weight] > 0.0) | (row[dobutamine_dose_weight] > 0)):
        #         val = 2
        # elif (row['best_map'] < 70):
        #     val = 1
        if ((row[epinephrine_dose_weight] > 0.0) & (row[norepinephrine_dose_weight] > 0.0)):
            val = 4
        elif ((row[epinephrine_dose_weight] > 0.0) | (row[norepinephrine_dose_weight] > 0.0)):
            val = 3
        elif ((row[dopamine_dose_weight] > 0.0) | (row[dobutamine_dose_weight] > 0)):
                val = 2
        elif (row['best_map'] < 70):
            val = 1
        elif (row['best_map'] >= 70):
            val = 0
        else:
            val = float("NaN")
        return val
    
    def SOFA_coag(self,
                  row):
        if row['platelets'] >= 150:
            val = 0
        elif (row['platelets'] >= 100) & (row['platelets'] < 150):
            val = 1
        elif (row['platelets'] >= 50) & (row['platelets'] < 100):
            val = 2
        elif (row['platelets'] >= 20) & (row['platelets'] < 50):
            val = 3
        elif (row['platelets'] < 20):
            val = 4
        else:
            val = float("NaN")
        return val
    
    def SOFA_hep(self,
                  row):
        if (row['bilirubin_total'] < 1.2):
            val = 0
        elif (row['bilirubin_total'] >= 1.2) & (row['bilirubin_total'] < 2.0):
            val = 1
        elif (row['bilirubin_total'] >= 2.0) & (row['bilirubin_total'] < 6.0):
            val = 2
        elif (row['bilirubin_total'] >= 6.0) & (row['bilirubin_total'] < 12.0):
            val = 3
        elif (row['bilirubin_total'] >= 12.0):
            val = 4
        else:
            val = float("NaN")
        return val
    
    def SOFA_renal(self,
                  row):
        if (row['creatinine'] < 1.2):
            val = 0
        elif (row['creatinine'] >= 1.2) & (row['creatinine'] < 2.0):
            val = 1
        elif (row['creatinine'] >= 2.0) & (row['creatinine'] < 3.5):
            val = 2
        elif (row['creatinine'] >= 3.5) & (row['creatinine'] < 5.0):
            val = 3
        elif (row['creatinine'] >= 5.0):
            val = 4
        else:
            val = float("NaN")
        return val
    
    def calculate_single_score(self, row: pd.Series) -> pd.DataFrame:
        """Calculate SOFA score for a single row."""
        scores = {}
        scores["SOFA_coag"] = self.SOFA_coag(row)
        scores["SOFA_renal"] = self.SOFA_renal(row)
        scores["SOFA_hep"] = self.SOFA_hep(row)
        scores["SOFA_neuro"] = self.SOFA_neuro(row)
        scores["SOFA_cardio"] = self.SOFA_cardio(row)
        scores["SOFA_cardio_mod"] = self.SOFA_cardio_mod(row)
        scores["SOFA_resp"] = self.SOFA_resp(row, column_name='p2f_vent_fio2')
        scores["SOFA_resp_sa"] = self.SOFA_resp(row, column_name='s2f_vent_fio2')
        scores["hourly_total"] = scores["SOFA_coag"] + scores["SOFA_renal"] + scores["SOFA_hep"] + scores["SOFA_neuro"] + scores["SOFA_cardio"] + scores["SOFA_resp"]

        scores_df = pd.DataFrame([scores])
        return scores_df

    def calculate_scores(self,
                         df: pd.DataFrame,
                         window = 24):
        """
        Calculates the Sequential Organ Failure Assessment (SOFA) score for a patient based on various organ systems.
        
        Args:
            window (int, optional): The rolling window size (in hours) used for calculating the delta of the SOFA score. The default value is 24 hours.
        """
    
        sofa_df = pd.DataFrame(index = df.index,
                               columns=[
                               'SOFA_coag',
                               'SOFA_renal',
                               'SOFA_hep',
                               'SOFA_neuro',
                               'SOFA_cardio',
                               'SOFA_cardio_mod',
                               'SOFA_resp',
                               'SOFA_resp_sa'])
        
        sofa_df['SOFA_coag'] = df.apply(self.SOFA_coag, axis=1)
        sofa_df['SOFA_renal'] = df.apply(self.SOFA_renal, axis=1)
        sofa_df['SOFA_hep'] = df.apply(self.SOFA_hep, axis=1)
        sofa_df['SOFA_neuro'] = df.apply(self.SOFA_neuro, axis=1)
        sofa_df['SOFA_cardio'] = df.apply(self.SOFA_cardio, axis=1)
        sofa_df['SOFA_cardio_mod'] = df.apply(self.SOFA_cardio_mod, axis=1)        
        sofa_df['SOFA_resp'] = df.apply(self.SOFA_resp, column_name='p2f_vent_fio2', axis=1)
        sofa_df['SOFA_resp_sa'] = df.apply(self.SOFA_resp, column_name='s2f_vent_fio2', axis=1)

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
        
        return sofa_df

class SIRSCalculator(ScoreCalculatorBase):
    """SIRS Score Calculator implementation."""
    
    def __init__(self, config: dict):
        self.config = config
        self.components = ['SIRS_resp', 'SIRS_cardio', 'SIRS_temp', 'SIRS_wbc']
        self.temperature_in_celsius = config['temperature_in_celsius']
    
    def SIRS_resp(self,
                  row,
                  resp_rate = 'unassisted_resp_rate',
                  paco2 = 'partial_pressure_of_carbon_dioxide_(paco2)'):
        """
        Accepts- class instance, one row from "super_table", "resp" cols
        Does- Calculates Respiratory SIRS score
        Returns- Single value of Respiratory SIRS score
        """
        if row[resp_rate] > 20:
            val = 1
        elif row[paco2] < 32:
            val = 1
        else: 
            val = 0
        return val

    def SIRS_cardio(self,
                  row,
                  hr = 'pulse'):
        """
        Accepts- class instance, one row from "super_table", "hr" cols
        Does- Calculates Cardiac SIRS score
        Returns- Single value of Cardiac SIRS score
        """
        if row[hr] > 90:
            val = 1
        else: 
            val = 0
        return val
    
    def SIRS_temp(self,
                  row,
                  temp = 'temperature'):
        """
        Accepts- class instance, one row from "super_table", "temp" cols
        Does- Calculates Temp SIRS score
        Returns- Single value of Temp SIRS score
        """
        if self.temperature_in_celsius:
            temp_high = 38.0
            temp_low = 36.0
        else:
            temp_high = 100.4
            temp_low = 95.8
            
        if row[temp] > temp_high:
            val = 1
        elif row[temp] < temp_low:
            val = 1
        else: 
            val = 0
        return val

    def SIRS_wbc(self,
                  row,
                  wbc = 'white_blood_cell_count'):
        """
        Accepts- class instance, one row from "super_table", "wbc" cols
        Does- Calculates White Blood Cell Count SIRS score
        Returns- Single value of White Blood Cell Count SIRS score
        """
        if row[wbc] > 12.0:
            val = 1
        elif row[wbc] < 4.0:
            val = 1
# =============================================================================
#         ## for bands
#         if row[bands] > 10:
#             val = 1
# =============================================================================
        else: 
            val = 0
        return val
    
    def calculate_scores(self,
                         df: pd.DataFrame,
                window = 24):
        """
        Calculates the SIRS (Systemic Inflammatory Response Syndrome) scores for a patient based on
        multiple physiological parameters over time.
        Args:
            window (int): The number of hours over which the rolling calculations are performed 
                                 (default is 24 hours). This affects the SIRS delta calculation and the 
                                 rolling total of the SIRS score.
        """

        sirs_df = pd.DataFrame(index = df.index,
                               columns=[
                               'SIRS_resp',
                               'SIRS_cardio',
                               'SIRS_temp',
                               'SIRS_wbc'])
        
        sirs_df['SIRS_resp'] = df.apply(self.SIRS_resp, axis=1)
        sirs_df['SIRS_cardio'] = df.apply(self.SIRS_cardio, axis=1)
        sirs_df['SIRS_temp'] = df.apply(self.SIRS_temp, axis=1)
        sirs_df['SIRS_wbc'] = df.apply(self.SIRS_wbc, axis=1)
                
        # Calculate hourly totals for each row
        sirs_df['hourly_total'] = sirs_df.sum(axis=1)
    
        # Calculate POST 24hr delta in total SIRS Score
        sirs_df['delta_24h'] = sirs_df['hourly_total'].\
        rolling(window=window, min_periods=24).\
        apply(lambda x: x.max() - x.min() if x.idxmax().value> x.idxmin().value else 0 ).tolist()
 
        # Calculate FIRST 24h delat in total SOFA score
        sirs_df.update(sirs_df.loc[sirs_df.index[0:24],['hourly_total']].\
        rolling(window=window, min_periods=1).max().rename(columns={'hourly_total':'delta_24h'}))
                
        return sirs_df
    
    def calculate_single_score(self, row: pd.Series) -> pd.DataFrame:
        """Calculate total SIRS score for a single row."""
        scores = {}
        scores["SIRS_resp"] = self.SIRS_resp(row)
        scores["SIRS_cardio"] = self.SIRS_cardio(row)
        scores["SIRS_temp"] = self.SIRS_temp(row)
        scores["SIRS_wbc"] = self.SIRS_wbc(row)
        scores["hourly_total"] = scores["SIRS_resp"] + scores["SIRS_cardio"] + scores["SIRS_temp"] + scores["SIRS_wbc"]
        scores_df = pd.DataFrame([scores])
        return scores_df
    
    def get_score_components(self) -> List[str]:
        """Return SIRS component names."""
        return self.components.copy()
    
    


class QSOFACalculator(ScoreCalculatorBase):
    """qSOFA Score Calculator implementation."""
    
    def __init__(self, config: dict):
        self.config = config
        self.components = ['qSOFA_resp', 'qSOFA_neuro', 'qSOFA_cardio']
    
    def calculate_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all qSOFA component scores for DataFrame."""
        qsofa_df = pd.DataFrame(index=df.index)
        
        # Respiratory qSOFA (RR ≥ 22)
        qsofa_df['qSOFA_resp'] = (df['unassisted_resp_rate'] >= 22).astype('int8')
        
        # Neurological qSOFA (GCS < 15)
        qsofa_df['qSOFA_neuro'] = (df['gcs_total_score'] < 15).astype('int8')
        
        # Cardiovascular qSOFA (SBP ≤ 100)
        qsofa_df['qSOFA_cardio'] = ((df['sbp_line'] <= 100) | (df['sbp_cuff'] <= 100)).astype('int8')
        
        # Total qSOFA score
        qsofa_df['total_score'] = qsofa_df[self.components].sum(axis=1).astype('int8')
        
        return qsofa_df
    
    def calculate_single_score(self, row: pd.Series) -> float:
        """Calculate total qSOFA score for a single row."""
        components = [
            self._calculate_resp_single(row),
            self._calculate_neuro_single(row),
            self._calculate_cardio_single(row)
        ]
        return sum(comp for comp in components if not pd.isna(comp))
    
    def get_score_components(self) -> List[str]:
        """Return qSOFA component names."""
        return self.components.copy()
    
    def _calculate_resp_single(self, row: pd.Series) -> int:
        """Calculate respiratory qSOFA for single row."""
        rr_value = row.get('unassisted_resp_rate', np.nan)
        return 1 if pd.notna(rr_value) and rr_value >= 22 else 0
    
    def _calculate_neuro_single(self, row: pd.Series) -> int:
        """Calculate neurological qSOFA for single row."""
        gcs_value = row.get('gcs_total_score', np.nan)
        return 1 if pd.notna(gcs_value) and gcs_value < 15 else 0
    
    def _calculate_cardio_single(self, row: pd.Series) -> int:
        """Calculate cardiovascular qSOFA for single row."""
        sbp_line = row.get('sbp_line', np.nan)
        sbp_cuff = row.get('sbp_cuff', np.nan)
        
        if (pd.notna(sbp_line) and sbp_line <= 100) or (pd.notna(sbp_cuff) and sbp_cuff <= 100):
            return 1
        else:
            return 0


class OrganSystemScoreCalculator(ScoreCalculatorBase):
    """Organ System Score Calculator implementation."""
    
    def __init__(self, config: dict):
        self.config = config
        self.components = ['MELD_score', 'AKI_score']
        self.aki_flagger = AKIFlagger()
    
    def get_score_components(self) -> List[str]:
        """Return organ system score component names."""
        return self.components.copy()
    
    def calculate_aki_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate AKI score for DataFrame."""
        dfin = pd.DataFrame(columns = ['patient_id', 'time', 'inpatient', 'creatinine'])
        dfin['creatinine'] = df['creatinine'].ffill().fillna(1)
        dfin['patient_id'] = [0]*len(dfin) #placeholder
        dfin['time'] = df.index
        dfin['inpatient'] = [True]*len(dfin)
        dfin['age'] = df['age']
        dfin['sex'] = df['gender']
        with warnings.catch_warnings():
            warnings.simplefilter(action = 'ignore', category = FutureWarning)
            out = self.aki_flagger.returnAKIpatients(dfin)
            aki_column = out['aki']
        
        aki_df = pd.DataFrame(index = df.index, columns = ['AKI_score'])
        aki_df['aki_score'] = aki_column
        return aki_df
    
    def calculate_meld_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MELD score for DataFrame."""
        meld_df = pd.DataFrame(index = df.index, columns = ['meld_score'])
        df_ = df.copy()
        df_.loc[df_.creatinine < 1, 'creatinine'] = 1
        df_.loc[df_.bilirubin_total < 1, 'bilirubin_total'] = 1
        df_.loc[df_.inr < 1, 'inr'] = 1
        
        meld = (np.log(df_.creatinine.ffill().fillna(1))*9.57) + \
		(3.78 * np.log(df_.bilirubin_total.ffill().fillna(1))) \
		+ (11.2 * np.log(df_.inr.ffill().fillna(1)))  + 6.43
        meld_df['meld_score'] = meld
        return meld_df
    
    def calculate_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all organ system scores for DataFrame."""
        organ_system_df = pd.DataFrame(index=df.index)
        organ_system_df['meld_score']  = self.calculate_meld_score(df)
        organ_system_df['aki_score'] = self.calculate_aki_score(df)
        return organ_system_df
    
    def calculate_single_score(self, row: pd.Series) -> pd.DataFrame:
        """Calculate all organ system scores for single row."""
        scores = {}
        scores["meld_score"] = self.calculate_meld_score(row)
        scores["aki_score"] = self.calculate_aki_score(row)
        scores_df = pd.DataFrame([scores])
        return scores_df
        
    