from enum import Enum
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Clinical score constants - moved from sepyDICT.py
RESAMPLE_FREQUENCY = '60min'
DEFAULT_WEIGHT_MALE = 89.0
DEFAULT_WEIGHT_FEMALE = 75.0
DEFAULT_HEIGHT_MALE = 175.3
DEFAULT_HEIGHT_FEMALE = 161.5
GENDER_MALE = 2
GENDER_FEMALE = 1

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

# Data validation constants
MAX_WEIGHT = 450.0
MIN_WEIGHT = 25.0
MIN_HEIGHT = 0.0
MIN_MAP = 30.0
MAX_MAP = 150.0


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


class ScoreType(Enum):
    """Enumeration of available score types."""
    SOFA = "sofa"
    SIRS = "sirs"
    APACHE = "apache"
    QSOFA = "qsofa"


class ScoreCalculatorBase(ABC):
    """Abstract base class for all score calculators."""
    
    def __init__(self, config: SepyDictConfig):
        self.config = config
    
    @abstractmethod
    def calculate_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate scores for the entire DataFrame."""
        pass
    
    @abstractmethod
    def calculate_single_score(self, row: pd.Series) -> float:
        """Calculate score for a single row."""
        pass
    
    @abstractmethod
    def get_score_components(self) -> List[str]:
        """Return list of score component names."""
        pass


class SOFACalculator(ScoreCalculatorBase):
    """SOFA Score Calculator implementation."""
    
    def __init__(self, config: SepyDictConfig):
        super().__init__(config)
        self.components = [
            'SOFA_resp', 'SOFA_cardio', 'SOFA_coag', 
            'SOFA_neuro', 'SOFA_hep', 'SOFA_renal'
        ]
    
    def calculate_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all SOFA component scores for DataFrame."""
        sofa_df = pd.DataFrame(index=df.index)
        
        # Use vectorized calculations
        sofa_df['SOFA_resp'] = self._calculate_resp_vectorized(df)
        sofa_df['SOFA_cardio'] = self._calculate_cardio_vectorized(df)
        sofa_df['SOFA_coag'] = self._calculate_coag_vectorized(df)
        sofa_df['SOFA_neuro'] = self._calculate_neuro_vectorized(df)
        sofa_df['SOFA_hep'] = self._calculate_hep_vectorized(df)
        sofa_df['SOFA_renal'] = self._calculate_renal_vectorized(df)
        
        # Calculate total score
        sofa_df['total_score'] = sofa_df[self.components].sum(axis=1)
        
        return sofa_df
    
    def calculate_single_score(self, row: pd.Series) -> float:
        """Calculate total SOFA score for a single row."""
        components = [
            self._calculate_resp_single(row),
            self._calculate_cardio_single(row),
            self._calculate_coag_single(row),
            self._calculate_neuro_single(row),
            self._calculate_hep_single(row),
            self._calculate_renal_single(row)
        ]
        return sum(comp for comp in components if not pd.isna(comp))
    
    def get_score_components(self) -> List[str]:
        """Return SOFA component names."""
        return self.components.copy()
    
    def _calculate_resp_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """Vectorized respiratory SOFA calculation."""
        conditions = [
            df['pf_pa'] < SOFA_PF_THRESHOLDS[3],  # < 100
            df['pf_pa'] < SOFA_PF_THRESHOLDS[2],  # < 200
            df['pf_pa'] < SOFA_PF_THRESHOLDS[1],  # < 300
            df['pf_pa'] < SOFA_PF_THRESHOLDS[0],  # < 400
        ]
        choices = [4, 3, 2, 1]
        return np.select(conditions, choices, default=0).astype('int8')
    
    def _calculate_cardio_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """Vectorized cardiovascular SOFA calculation."""
        dopamine = df['dopamine_dose_weight'].fillna(0)
        epinephrine = df['epinephrine_dose_weight'].fillna(0)
        norepinephrine = df['norepinephrine_dose_weight'].fillna(0)
        dobutamine = df['dobutamine_dose_weight'].fillna(0)
        
        conditions = [
            (dopamine > DOPAMINE_HIGH_THRESHOLD) | 
            (epinephrine > EPINEPHRINE_HIGH_THRESHOLD) | 
            (norepinephrine > NOREPINEPHRINE_HIGH_THRESHOLD),
            
            (dopamine > DOPAMINE_MID_THRESHOLD) |
            ((epinephrine > EPINEPHRINE_LOW_THRESHOLD) & (epinephrine <= EPINEPHRINE_HIGH_THRESHOLD)) |
            ((norepinephrine > NOREPINEPHRINE_LOW_THRESHOLD) & (norepinephrine <= NOREPINEPHRINE_HIGH_THRESHOLD)),
            
            ((dopamine > DOPAMINE_LOW_THRESHOLD) & (dopamine <= DOPAMINE_MID_THRESHOLD)) | 
            (dobutamine > DOBUTAMINE_LOW_THRESHOLD),
            
            df['best_map'] < MAP_THRESHOLD
        ]
        
        choices = [4, 3, 2, 1]
        return np.select(conditions, choices, default=0).astype('int8')
    
    def _calculate_coag_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """Vectorized coagulation SOFA calculation."""
        conditions = [
            df['platelets'] < SOFA_PLATELETS_THRESHOLDS[3],  # < 20
            df['platelets'] < SOFA_PLATELETS_THRESHOLDS[2],  # < 50
            df['platelets'] < SOFA_PLATELETS_THRESHOLDS[1],  # < 100
            df['platelets'] < SOFA_PLATELETS_THRESHOLDS[0],  # < 150
        ]
        choices = [4, 3, 2, 1]
        return np.select(conditions, choices, default=0).astype('int8')
    
    def _calculate_neuro_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """Vectorized neurological SOFA calculation."""
        conditions = [
            df['gcs_total_score'] < SOFA_GCS_THRESHOLDS[3],  # < 6
            df['gcs_total_score'] < SOFA_GCS_THRESHOLDS[2],  # < 10
            df['gcs_total_score'] < SOFA_GCS_THRESHOLDS[1],  # < 13
            df['gcs_total_score'] < SOFA_GCS_THRESHOLDS[0],  # < 15
        ]
        choices = [4, 3, 2, 1]
        return np.select(conditions, choices, default=0).astype('int8')
    
    def _calculate_hep_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """Vectorized hepatic SOFA calculation."""
        conditions = [
            df['bilirubin_total'] >= SOFA_BILIRUBIN_THRESHOLDS[3],  # >= 12.0
            df['bilirubin_total'] >= SOFA_BILIRUBIN_THRESHOLDS[2],  # >= 6.0
            df['bilirubin_total'] >= SOFA_BILIRUBIN_THRESHOLDS[1],  # >= 2.0
            df['bilirubin_total'] >= SOFA_BILIRUBIN_THRESHOLDS[0],  # >= 1.2
        ]
        choices = [4, 3, 2, 1]
        return np.select(conditions, choices, default=0).astype('int8')
    
    def _calculate_renal_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """Vectorized renal SOFA calculation."""
        conditions = [
            df['creatinine'] >= SOFA_CREATININE_THRESHOLDS[3],  # >= 5.0
            df['creatinine'] >= SOFA_CREATININE_THRESHOLDS[2],  # >= 3.5
            df['creatinine'] >= SOFA_CREATININE_THRESHOLDS[1],  # >= 2.0
            df['creatinine'] >= SOFA_CREATININE_THRESHOLDS[0],  # >= 1.2
        ]
        choices = [4, 3, 2, 1]
        return np.select(conditions, choices, default=0).astype('int8')
    
    # Single row calculation methods
    def _calculate_resp_single(self, row: pd.Series) -> float:
        """Calculate respiratory SOFA for single row."""
        pf_value = row.get('pf_pa', np.nan)
        if pd.isna(pf_value):
            return np.nan
        
        if pf_value < SOFA_PF_THRESHOLDS[3]:
            return 4
        elif pf_value < SOFA_PF_THRESHOLDS[2]:
            return 3
        elif pf_value < SOFA_PF_THRESHOLDS[1]:
            return 2
        elif pf_value < SOFA_PF_THRESHOLDS[0]:
            return 1
        else:
            return 0
    
    def _calculate_cardio_single(self, row: pd.Series) -> float:
        """Calculate cardiovascular SOFA for single row."""
        dopamine = row.get('dopamine_dose_weight', 0) or 0
        epinephrine = row.get('epinephrine_dose_weight', 0) or 0
        norepinephrine = row.get('norepinephrine_dose_weight', 0) or 0
        dobutamine = row.get('dobutamine_dose_weight', 0) or 0
        best_map = row.get('best_map', np.nan)
        
        if ((dopamine > DOPAMINE_HIGH_THRESHOLD) |
            (epinephrine > EPINEPHRINE_HIGH_THRESHOLD) | 
            (norepinephrine > NOREPINEPHRINE_HIGH_THRESHOLD)):
            return 4
        elif ((dopamine > DOPAMINE_MID_THRESHOLD) |
              ((epinephrine > EPINEPHRINE_LOW_THRESHOLD) & (epinephrine <= EPINEPHRINE_HIGH_THRESHOLD)) | 
              ((norepinephrine > NOREPINEPHRINE_LOW_THRESHOLD) & (norepinephrine <= NOREPINEPHRINE_HIGH_THRESHOLD))):
            return 3
        elif (((dopamine > DOPAMINE_LOW_THRESHOLD) & (dopamine <= DOPAMINE_MID_THRESHOLD)) |
              (dobutamine > DOBUTAMINE_LOW_THRESHOLD)):
            return 2
        elif pd.notna(best_map) and (best_map < MAP_THRESHOLD):
            return 1
        elif pd.notna(best_map) and (best_map >= MAP_THRESHOLD):
            return 0
        else:
            return np.nan
    
    def _calculate_coag_single(self, row: pd.Series) -> float:
        """Calculate coagulation SOFA for single row."""
        platelets = row.get('platelets', np.nan)
        if pd.isna(platelets):
            return np.nan
        
        if platelets >= SOFA_PLATELETS_THRESHOLDS[0]:
            return 0
        elif platelets >= SOFA_PLATELETS_THRESHOLDS[1]:
            return 1
        elif platelets >= SOFA_PLATELETS_THRESHOLDS[2]:
            return 2
        elif platelets >= SOFA_PLATELETS_THRESHOLDS[3]:
            return 3
        else:
            return 4
    
    def _calculate_neuro_single(self, row: pd.Series) -> float:
        """Calculate neurological SOFA for single row."""
        gcs = row.get('gcs_total_score', np.nan)
        if pd.isna(gcs):
            return np.nan
        
        if gcs == SOFA_GCS_THRESHOLDS[0]:
            return 0
        elif gcs >= SOFA_GCS_THRESHOLDS[1]:
            return 1
        elif gcs >= SOFA_GCS_THRESHOLDS[2]:
            return 2
        elif gcs >= SOFA_GCS_THRESHOLDS[3]:
            return 3
        else:
            return 4
    
    def _calculate_hep_single(self, row: pd.Series) -> float:
        """Calculate hepatic SOFA for single row."""
        bilirubin = row.get('bilirubin_total', np.nan)
        if pd.isna(bilirubin):
            return np.nan
        
        if bilirubin < SOFA_BILIRUBIN_THRESHOLDS[0]:
            return 0
        elif bilirubin < SOFA_BILIRUBIN_THRESHOLDS[1]:
            return 1
        elif bilirubin < SOFA_BILIRUBIN_THRESHOLDS[2]:
            return 2
        elif bilirubin < SOFA_BILIRUBIN_THRESHOLDS[3]:
            return 3
        else:
            return 4
    
    def _calculate_renal_single(self, row: pd.Series) -> float:
        """Calculate renal SOFA for single row."""
        creatinine = row.get('creatinine', np.nan)
        if pd.isna(creatinine):
            return np.nan
        
        if creatinine < SOFA_CREATININE_THRESHOLDS[0]:
            return 0
        elif creatinine < SOFA_CREATININE_THRESHOLDS[1]:
            return 1
        elif creatinine < SOFA_CREATININE_THRESHOLDS[2]:
            return 2
        elif creatinine < SOFA_CREATININE_THRESHOLDS[3]:
            return 3
        else:
            return 4


class SIRSCalculator(ScoreCalculatorBase):
    """SIRS Score Calculator implementation."""
    
    def __init__(self, config: SepyDictConfig):
        super().__init__(config)
        self.components = ['SIRS_resp', 'SIRS_cardio', 'SIRS_temp', 'SIRS_wbc']
    
    def calculate_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all SIRS component scores for DataFrame."""
        sirs_df = pd.DataFrame(index=df.index)
        
        # Respiratory SIRS (vectorized)
        sirs_df['SIRS_resp'] = ((df['unassisted_resp_rate'] > RESP_RATE_THRESHOLD) | 
                               (df['partial_pressure_of_carbon_dioxide_(paco2)'] < PACO2_THRESHOLD)).astype('int8')
        
        # Cardiac SIRS (vectorized)
        sirs_df['SIRS_cardio'] = (df['pulse'] > HEART_RATE_THRESHOLD).astype('int8')
        
        # Temperature SIRS (vectorized)
        sirs_df['SIRS_temp'] = ((df['temperature'] > TEMPERATURE_HIGH_F) | 
                               (df['temperature'] < TEMPERATURE_LOW_F)).astype('int8')
        
        # WBC SIRS (vectorized)
        sirs_df['SIRS_wbc'] = ((df['white_blood_cell_count'] > WBC_HIGH_THRESHOLD) | 
                              (df['white_blood_cell_count'] < WBC_LOW_THRESHOLD)).astype('int8')
        
        # Total SIRS score
        sirs_df['total_score'] = sirs_df[self.components].sum(axis=1).astype('int8')
        
        return sirs_df
    
    def calculate_single_score(self, row: pd.Series) -> float:
        """Calculate total SIRS score for a single row."""
        components = [
            self._calculate_resp_single(row),
            self._calculate_cardio_single(row),
            self._calculate_temp_single(row),
            self._calculate_wbc_single(row)
        ]
        return sum(comp for comp in components if not pd.isna(comp))
    
    def get_score_components(self) -> List[str]:
        """Return SIRS component names."""
        return self.components.copy()
    
    def _calculate_resp_single(self, row: pd.Series) -> int:
        """Calculate respiratory SIRS for single row."""
        rr_value = row.get('unassisted_resp_rate', np.nan)
        paco2_value = row.get('partial_pressure_of_carbon_dioxide_(paco2)', np.nan)
        
        if (pd.notna(rr_value) and rr_value > RESP_RATE_THRESHOLD) or \
           (pd.notna(paco2_value) and paco2_value < PACO2_THRESHOLD):
            return 1
        else: 
            return 0
    
    def _calculate_cardio_single(self, row: pd.Series) -> int:
        """Calculate cardiac SIRS for single row."""
        hr_value = row.get('pulse', np.nan)
        if pd.notna(hr_value) and hr_value > HEART_RATE_THRESHOLD:
            return 1
        else: 
            return 0
    
    def _calculate_temp_single(self, row: pd.Series) -> int:
        """Calculate temperature SIRS for single row."""
        temp_value = row.get('temperature', np.nan)
        if pd.notna(temp_value) and (temp_value > TEMPERATURE_HIGH_F or temp_value < TEMPERATURE_LOW_F):
            return 1
        else: 
            return 0
    
    def _calculate_wbc_single(self, row: pd.Series) -> int:
        """Calculate WBC SIRS for single row."""
        wbc_value = row.get('white_blood_cell_count', np.nan)
        if pd.notna(wbc_value) and (wbc_value > WBC_HIGH_THRESHOLD or wbc_value < WBC_LOW_THRESHOLD):
            return 1
        else: 
            return 0


class QSOFACalculator(ScoreCalculatorBase):
    """qSOFA Score Calculator implementation."""
    
    def __init__(self, config: SepyDictConfig):
        super().__init__(config)
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


class ScoreCalculatorFactory:
    """Factory class for creating score calculators."""
    
    _calculators = {
        ScoreType.SOFA: SOFACalculator,
        ScoreType.SIRS: SIRSCalculator,
        ScoreType.QSOFA: QSOFACalculator,
    }
    
    @classmethod
    def create_calculator(cls, score_type: ScoreType, config: SepyDictConfig) -> ScoreCalculatorBase:
        """
        Create a score calculator instance.
        
        Args:
            score_type: Type of score calculator to create
            config: Configuration object
            
        Returns:
            Score calculator instance
            
        Raises:
            ValueError: If score_type is not supported
        """
        if score_type not in cls._calculators:
            available_types = list(cls._calculators.keys())
            raise ValueError(f"Unsupported score type: {score_type}. Available types: {available_types}")
        
        calculator_class = cls._calculators[score_type]
        return calculator_class(config)
    
    @classmethod
    def register_calculator(cls, score_type: ScoreType, calculator_class: type) -> None:
        """
        Register a new score calculator type.
        
        Args:
            score_type: Type identifier for the calculator
            calculator_class: Calculator class that inherits from ScoreCalculatorBase
        """
        if not issubclass(calculator_class, ScoreCalculatorBase):
            raise ValueError("Calculator class must inherit from ScoreCalculatorBase")
        
        cls._calculators[score_type] = calculator_class
    
    @classmethod
    def get_available_types(cls) -> List[ScoreType]:
        """Get list of available score calculator types."""
        return list(cls._calculators.keys())
    
    @classmethod
    def create_all_calculators(cls, config: SepyDictConfig) -> Dict[ScoreType, ScoreCalculatorBase]:
        """
        Create instances of all available calculators.
        
        Args:
            config: Configuration object
            
        Returns:
            Dictionary mapping score types to calculator instances
        """
        return {
            score_type: cls.create_calculator(score_type, config)
            for score_type in cls._calculators.keys()
        }
