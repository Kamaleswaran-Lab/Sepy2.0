"""
Fluid Processing Module

This module processes fluid administration data from clinical descriptions,
extracting volume, rate, and time information to create structured records.

Author: Refactored for improved efficiency and best practices
"""

import re
import logging
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FluidParameters:
    """Data class to hold extracted fluid parameters."""
    bolus: Optional[float] = None
    volume: Optional[float] = None
    rate: Optional[float] = None
    duration: Optional[float] = None
    
    def is_empty(self) -> bool:
        """Check if all parameters are None."""
        return all(param is None for param in [self.bolus, self.volume, self.rate, self.duration])
    

# Configuration class for better maintainability
@dataclass
class FluidProcessorConfig:
    """Configuration for FluidProcessor."""
    max_duration_hours: float = 1000.0  # Maximum reasonable duration
    max_rate_ml_hr: float = 20000.0   # Maximum reasonable rate
    max_volume_ml: float = 10000.0     # Maximum reasonable volume
    min_positive_value: float = 0.01  # Minimum positive value to consider
    enable_validation: bool = True     # Enable parameter validation
    
    def validate_parameters(self, params: FluidParameters) -> FluidParameters:
        """Validate and clean parameters based on configuration."""
        if not self.enable_validation:
            return params
            
        # Validate and cap values
        if params.duration and params.duration > self.max_duration_hours:
            logger.warning(f"Duration {params.duration}h exceeds maximum, capping at {self.max_duration_hours}h")
            params.duration = self.max_duration_hours
            
        if params.rate and params.rate > self.max_rate_ml_hr:
            logger.warning(f"Rate {params.rate} mL/hr exceeds maximum, capping at {self.max_rate_ml_hr}")
            params.rate = self.max_rate_ml_hr
            
        if params.volume and params.volume > self.max_volume_ml:
            logger.warning(f"Volume {params.volume} mL exceeds maximum, capping at {self.max_volume_ml}")
            params.volume = self.max_volume_ml
            
        # Remove values that are too small to be meaningful
        if params.duration and params.duration < self.min_positive_value:
            params.duration = None
        if params.rate and params.rate < self.min_positive_value:
            params.rate = None
        if params.volume and params.volume < self.min_positive_value:
            params.volume = None
        if params.bolus and params.bolus < self.min_positive_value:
            params.bolus = None
            
        return params


@dataclass
class ProcessingStats:
    """Statistics tracker for fluid processing."""
    total_records: int = 0
    none_found: int = 0
    only_time: int = 0
    only_volume: int = 0
    only_rate: int = 0
    volume_rate: int = 0
    volume_time: int = 0
    rate_time: int = 0
    volume_rate_time: int = 0
    errors: int = 0
    processed_successfully: int = 0

    def update_stats(self, params: FluidParameters) -> None:
        """Update statistics based on extracted parameters."""
        has_vol = params.volume is not None
        has_rate = params.rate is not None
        has_time = params.duration is not None
        
        if not has_vol and not has_rate and not has_time:
            self.none_found += 1
        elif not has_vol and not has_rate and has_time:
            self.only_time += 1
        elif has_vol and not has_rate and not has_time:
            self.only_volume += 1
        elif not has_vol and has_rate and not has_time:
            self.only_rate += 1
        elif has_vol and has_rate and not has_time:
            self.volume_rate += 1
        elif has_vol and not has_rate and has_time:
            self.volume_time += 1
        elif not has_vol and has_rate and has_time:
            self.rate_time += 1
        elif has_vol and has_rate and has_time:
            self.volume_rate_time += 1

    def print_summary(self) -> None:
        """Print processing statistics summary."""
        if self.total_records == 0:
            logger.warning("No records processed")
            return
            
        stats_text = f"""
        Processing Summary:
        ==================
        Total Records: {self.total_records}
        Successfully Processed: {self.processed_successfully} ({self.processed_successfully/self.total_records*100:.1f}%)
        Errors: {self.errors} ({self.errors/self.total_records*100:.1f}%)
        
        Parameter Combinations:
        - None found: {self.none_found} ({self.none_found/self.total_records*100:.1f}%)
        - Only volume: {self.only_volume} ({self.only_volume/self.total_records*100:.1f}%)
        - Only rate: {self.only_rate} ({self.only_rate/self.total_records*100:.1f}%)
        - Only time: {self.only_time} ({self.only_time/self.total_records*100:.1f}%)
        - Volume + Rate: {self.volume_rate} ({self.volume_rate/self.total_records*100:.1f}%)
        - Volume + Time: {self.volume_time} ({self.volume_time/self.total_records*100:.1f}%)
        - Rate + Time: {self.rate_time} ({self.rate_time/self.total_records*100:.1f}%)
        - All three: {self.volume_rate_time} ({self.volume_rate_time/self.total_records*100:.1f}%)
        """
        logger.info(stats_text)


class FluidProcessor:
    """Comprehensive fluid processing class with all features integrated."""
    
    # Compile regex patterns once for better performance
    NUMBER_PATTERN = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")
    BOLUS_PATTERN = re.compile(r'\bbolus\b', re.IGNORECASE)
    VOLUME_PATTERN = re.compile(r'\bmL\b(?!/hr)', re.IGNORECASE)
    RATE_PATTERN = re.compile(r'\bmL/hr\b', re.IGNORECASE)
    TIME_HR_PATTERN = re.compile(r'\bhr\(s\)?\b', re.IGNORECASE)
    TIME_MIN_PATTERN = re.compile(r'\bminute\(s\)?\b', re.IGNORECASE)
    
    def __init__(self, config: Optional[FluidProcessorConfig] = None):
        self.stats = ProcessingStats()
        self.config = config or FluidProcessorConfig()
    
    def extract_numbers(self, text_list: List[str]) -> Optional[float]:
        """
        Extract and return the maximum numeric value from a list of text strings.
        
        Args:
            text_list: List of strings potentially containing numbers
            
        Returns:
            Maximum numeric value found or None if no valid numbers
        """
        if not text_list:
            return None
            
        try:
            # Remove commas and extract numbers
            cleaned_texts = [text.replace(',', '') for text in text_list]
            all_numbers = []
            
            for text in cleaned_texts:
                numbers = self.NUMBER_PATTERN.findall(text)
                all_numbers.extend([float(num) for num in numbers])
            
            if all_numbers:
                max_val = max(all_numbers)
                return max_val if max_val > 0 else None
                
        except (ValueError, TypeError) as e:
            logger.warning(f"Error extracting numbers from {text_list}: {e}")
            
        return None
    
    def parse_clinical_description(self, description: str) -> FluidParameters:
        """
        Parse clinical description to extract fluid parameters with validation.
        
        Args:
            description: Clinical description text
            
        Returns:
            FluidParameters object with extracted and validated values
        """
        if not isinstance(description, str):
            logger.warning(f"Invalid description type: {type(description)}")
            return FluidParameters()
        
        try:
            # Split description into components
            desc_parts = [part.strip() for part in description.split(', ')]
            
            params = FluidParameters()
            
            # Extract bolus
            bolus_parts = [part for part in desc_parts if self.BOLUS_PATTERN.search(part)]
            if bolus_parts:
                params.bolus = self.extract_numbers(bolus_parts)

            # Extract volume (mL but not mL/hr)
            volume_parts = []
            for part in desc_parts:
                if self.VOLUME_PATTERN.search(part) and not self.RATE_PATTERN.search(part):
                    volume_parts.append(part)
            
            if volume_parts:
                params.volume = self.extract_numbers(volume_parts)
                    
            # Extract rate (mL/hr)
            rate_parts = [part for part in desc_parts if self.RATE_PATTERN.search(part)]
            if rate_parts:
                # Handle case where 'mL/hr' appears alone
                processed_rate_parts = []
                for i, part in enumerate(desc_parts):
                    if part.strip() == 'mL/hr' and i > 0:
                        processed_rate_parts.append(desc_parts[i-1])
                    elif self.RATE_PATTERN.search(part) and part.strip() != 'mL/hr':
                        processed_rate_parts.append(part)
                
                if processed_rate_parts:
                    params.rate = self.extract_numbers(processed_rate_parts)
            
            # Extract time duration
            time_parts = []
            for part in desc_parts:
                if self.TIME_HR_PATTERN.search(part):
                    time_parts.append(('hr', part))
                elif self.TIME_MIN_PATTERN.search(part):
                    time_parts.append(('min', part))
            
            if time_parts:
                time_unit, time_text = time_parts[0]  # Take first match
                duration_value = self.extract_numbers([time_text])
                if duration_value:
                    params.duration = duration_value if time_unit == 'hr' else duration_value / 60.0
            
            # Calculate missing parameters (like original code)
            try:
                if params.volume and params.rate and not params.duration:
                    params.duration = params.volume / params.rate
                elif params.volume and params.duration and not params.rate:
                    params.rate = params.volume / params.duration
                elif params.rate and params.duration and not params.volume:
                    params.volume = params.rate * params.duration
            except (ZeroDivisionError, TypeError) as e:
                logger.warning(f"Error calculating parameters: {e}")  
            
            # Apply configuration-based validation
            params = self.config.validate_parameters(params)
            
            return params
            
        except Exception as e:
            logger.error(f"Error parsing description '{description}': {e}")
            return FluidParameters()    

    
    def create_time_series_records(self, base_row: pd.DataFrame, params: FluidParameters) -> pd.DataFrame:
        """
        Create time-series records for continuous infusions.
        
        Args:
            base_row: Original dataframe row
            params: Fluid parameters with rate and duration
            
        Returns:
            DataFrame with time-series records
        """
        try:
            if not params.rate or not params.duration:
                return base_row
            
            # Calculate number of hourly intervals
            full_hours = int(params.duration)
            partial_hour = params.duration % 1
            
            records = []
            
            # Create records for full hours
            for hour in range(full_hours):
                record = base_row.copy()
                record.at[0, 'volume'] = params.rate
                record.at[0, 'service_ts'] = record.at[0, 'service_ts'] + pd.Timedelta(hours=hour)
                records.append(record)
            
            # Add partial hour record if needed
            if partial_hour > 0:
                record = base_row.copy()
                record.at[0, 'volume'] = params.rate * partial_hour
                record.at[0, 'service_ts'] = record.at[0, 'service_ts'] + pd.Timedelta(hours=full_hours)
                records.append(record)
            
            if records:
                return pd.concat(records, ignore_index=True)
            else:
                return base_row
                
        except Exception as e:
            logger.error(f"Error creating time series records: {e}")
            return base_row
    
    def process_single_record(self, row_data: Tuple[int, pd.Series]) -> Optional[pd.DataFrame]:
        """
        Process a single fluid record.
        
        Args:
            row_data: Tuple of (index, pandas Series)
            
        Returns:
            Processed DataFrame or None if processing failed
        """
        try:
            idx, row = row_data
            description = row.get('order_clinical_desc', '')
            
            if not description:
                logger.warning(f"Empty description for row {idx}")
                return None
            
            # Parse clinical description
            params = self.parse_clinical_description(description)
            
            # Update statistics
            self.stats.update_stats(params)
            
            # Skip if no useful parameters found
            if params.is_empty():
                return None
            
            # Create base dataframe row
            base_row = pd.DataFrame([row]).reset_index(drop=True)
            
            # Process based on available parameters (matching original logic exactly)
            if params.bolus:
                # Bolus administration
                base_row.at[0, 'volume'] = params.bolus
                output_df = base_row
                
            elif params.volume and not params.rate and not params.duration:
                # Simple volume administration (volume only)
                base_row.at[0, 'volume'] = params.volume
                output_df = base_row
                
            elif params.rate and params.duration:
                # Continuous infusion - create time series (rate + duration)
                output_df = self.create_time_series_records(base_row, params)
                
            else:
                return None
            
            return output_df
            
        except Exception as e:
            logger.error(f"Error processing record {idx}: {e}")
            self.stats.errors += 1
            return None
    
    def validate_input_data(self, fluid_data: pd.DataFrame) -> bool:
        """Validate input DataFrame structure and content."""
        required_columns = ['order_clinical_desc', 'service_ts']
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in fluid_data.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
            
        # Check for empty descriptions
        empty_desc_count = fluid_data['order_clinical_desc'].isna().sum()
        if empty_desc_count > 0:
            logger.warning(f"Found {empty_desc_count} empty clinical descriptions")
            
        return True

    def process_fluids(
        self, 
        fluid_data: pd.DataFrame, 
        batch_size: int = 1000
    ) -> Tuple[pd.DataFrame, ProcessingStats]:
        """
        Process fluid administration data with batch processing and validation.
        
        Args:
            fluid_data: DataFrame containing fluid administration data
            batch_size: Size of processing batches for memory management
            
        Returns:
            Tuple of (processed_dataframe, processing_stats)
        """
        # Validate inputs
        if fluid_data.empty:
            logger.error("Input DataFrame is empty")
            return pd.DataFrame(), self.stats
            
        if not self.validate_input_data(fluid_data):
            return pd.DataFrame(), self.stats
        
        # Initialize statistics
        self.stats.total_records = len(fluid_data)
        logger.info(f"Processing {self.stats.total_records} records in batches of {batch_size}")
        
        successful_count = 0
        processed_records = []
        
        # Process in batches for better memory management
        for i in range(0, len(fluid_data), batch_size):
            batch_end = min(i + batch_size, len(fluid_data))
            batch_data = fluid_data.iloc[i:batch_end]
            
            batch_num = i//batch_size + 1
            total_batches = (len(fluid_data)-1)//batch_size + 1
            logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            # Process batch
            for row_data in tqdm(batch_data.iterrows(), 
                               desc=f"Batch {batch_num}", 
                               total=len(batch_data)):
                output_df = self.process_single_record(row_data)
                if output_df is not None:
                    processed_records.append(output_df)
                    successful_count += 1
        
        self.stats.processed_successfully = successful_count
        
        # Combine all processed records
        if processed_records:
            final_df = pd.concat(processed_records, ignore_index=True)
        else:
            final_df = pd.DataFrame()
        
        # Print summary
        self.stats.print_summary()
        
        return final_df, self.stats
    
    def get_summary_dataframe(self) -> pd.DataFrame:
        """Get processing summary as a DataFrame."""
        summary_data = {
            'Metric': [
                'Total Records', 'Successfully Processed', 'Errors',
                'None Found', 'Only Volume', 'Only Rate', 'Only Time',
                'Volume + Rate', 'Volume + Time', 'Rate + Time', 'All Three'
            ],
            'Count': [
                self.stats.total_records, self.stats.processed_successfully, self.stats.errors,
                self.stats.none_found, self.stats.only_volume, self.stats.only_rate, 
                self.stats.only_time, self.stats.volume_rate, self.stats.volume_time,
                self.stats.rate_time, self.stats.volume_rate_time
            ],
            'Percentage': [
                100.0,
                self.stats.processed_successfully / self.stats.total_records * 100 if self.stats.total_records > 0 else 0,
                self.stats.errors / self.stats.total_records * 100 if self.stats.total_records > 0 else 0,
                self.stats.none_found / self.stats.total_records * 100 if self.stats.total_records > 0 else 0,
                self.stats.only_volume / self.stats.total_records * 100 if self.stats.total_records > 0 else 0,
                self.stats.only_rate / self.stats.total_records * 100 if self.stats.total_records > 0 else 0,
                self.stats.only_time / self.stats.total_records * 100 if self.stats.total_records > 0 else 0,
                self.stats.volume_rate / self.stats.total_records * 100 if self.stats.total_records > 0 else 0,
                self.stats.volume_time / self.stats.total_records * 100 if self.stats.total_records > 0 else 0,
                self.stats.rate_time / self.stats.total_records * 100 if self.stats.total_records > 0 else 0,
                self.stats.volume_rate_time / self.stats.total_records * 100 if self.stats.total_records > 0 else 0
            ]
        }
        
        return pd.DataFrame(summary_data)

    def process_fluids_vectorized(
        self, 
        fluid_data: pd.DataFrame, 
        batch_size: int = 10000
    ) -> Tuple[pd.DataFrame, ProcessingStats]:
        """
        More efficient vectorized processing of fluid administration data.
        
        Args:
            fluid_data: DataFrame containing fluid administration data
            batch_size: Size of processing batches for memory management
            
        Returns:
            Tuple of (processed_dataframe, processing_stats)
        """
        # Validate inputs
        if fluid_data.empty:
            logger.error("Input DataFrame is empty")
            return pd.DataFrame(), self.stats
            
        if not self.validate_input_data(fluid_data):
            return pd.DataFrame(), self.stats
        
        # Initialize statistics
        self.stats.total_records = len(fluid_data)
        logger.info(f"Processing {self.stats.total_records} records with vectorized approach")
        
        # Pre-process all clinical descriptions in bulk
        logger.info("Parsing clinical descriptions...")
        descriptions = fluid_data['order_clinical_desc'].fillna('')
        
        # Vectorized parameter extraction
        parsed_params = self._bulk_parse_descriptions(descriptions.tolist())
        
        # Create results DataFrame more efficiently
        processed_records = self._bulk_create_records(fluid_data, parsed_params)
        
        # Update statistics
        for params in parsed_params:
            self.stats.update_stats(params)
            if not params.is_empty():
                self.stats.processed_successfully += 1
        
        # Print summary
        self.stats.print_summary()
        
        return processed_records, self.stats
    
    def _bulk_parse_descriptions(self, descriptions: List[str]) -> List[FluidParameters]:
        """
        Parse multiple clinical descriptions efficiently using vectorized operations.
        """
        params_list = []
        
        # Use tqdm for progress tracking
        for desc in tqdm(descriptions, desc="Parsing descriptions"):
            params = self.parse_clinical_description(desc)
            params_list.append(params)
        
        return params_list
    
    def _bulk_create_records(self, fluid_data: pd.DataFrame, params_list: List[FluidParameters]) -> pd.DataFrame:
        """
        Create output records efficiently using bulk operations.
        """
        # Pre-filter valid records
        valid_indices = []
        record_data = []
        
        for idx, params in enumerate(params_list):
            if params.is_empty():
                continue
                
            row = fluid_data.iloc[idx]
            
            if params.bolus:
                # Simple bolus record
                record_data.append({
                    **row.to_dict(),
                    'volume': params.bolus,
                    'record_type': 'bolus'
                })
                valid_indices.append(idx)
                
            elif params.volume and not params.rate and not params.duration:
                # Simple volume record
                record_data.append({
                    **row.to_dict(),
                    'volume': params.volume,
                    'record_type': 'volume'
                })
                valid_indices.append(idx)
                
            elif params.rate and params.duration:
                # Time series records - this is more complex
                ts_records = self._create_time_series_bulk(row, params)
                record_data.extend(ts_records)
                valid_indices.extend([idx] * len(ts_records))
        
        # Create DataFrame from all records at once
        if record_data:
            return pd.DataFrame(record_data)
        else:
            return pd.DataFrame()
    
    def _create_time_series_bulk(self, base_row: pd.Series, params: FluidParameters) -> List[dict]:
        """
        Create time-series records more efficiently.
        """
        if not params.rate or not params.duration:
            return []
        
        records = []
        base_dict = base_row.to_dict()
        base_ts = base_row['service_ts']
        
        # Calculate intervals
        full_hours = int(params.duration)
        partial_hour = params.duration % 1
        
        # Create records for full hours
        for hour in range(full_hours):
            record = base_dict.copy()
            record['volume'] = params.rate
            record['service_ts'] = base_ts + pd.Timedelta(hours=hour)
            record['record_type'] = 'time_series'
            records.append(record)
        
        # Add partial hour record
        if partial_hour > 0:
            record = base_dict.copy()
            record['volume'] = params.rate * partial_hour
            record['service_ts'] = base_ts + pd.Timedelta(hours=full_hours)
            record['record_type'] = 'time_series_partial'
            records.append(record)
        
        return records

    def process_fluids_apply(
        self, 
        fluid_data: pd.DataFrame, 
        batch_size: int = 5000
    ) -> Tuple[pd.DataFrame, ProcessingStats]:
        """
        Alternative efficient processing using pandas apply (middle ground approach).
        
        Args:
            fluid_data: DataFrame containing fluid administration data
            batch_size: Size of processing batches
            
        Returns:
            Tuple of (processed_dataframe, processing_stats)
        """
        # Validate inputs
        if fluid_data.empty:
            logger.error("Input DataFrame is empty")
            return pd.DataFrame(), self.stats
            
        if not self.validate_input_data(fluid_data):
            return pd.DataFrame(), self.stats
        
        self.stats.total_records = len(fluid_data)
        logger.info(f"Processing {self.stats.total_records} records using apply method")
        
        # Process in batches
        all_results = []
        
        for i in range(0, len(fluid_data), batch_size):
            batch_end = min(i + batch_size, len(fluid_data))
            batch_data = fluid_data.iloc[i:batch_end].copy()
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(fluid_data)-1)//batch_size + 1}")
            
            # Use apply which is faster than iterrows
            results = batch_data.apply(self._process_row_apply, axis=1)
            
            # Filter out None results and extend list
            valid_results = [r for r in results if r is not None]
            all_results.extend(valid_results)
        
        # Concatenate all results
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
        else:
            final_df = pd.DataFrame()
        
        self.stats.processed_successfully = len(all_results)
        self.stats.print_summary()
        
        return final_df, self.stats
    
    def _process_row_apply(self, row: pd.Series) -> Optional[pd.DataFrame]:
        """
        Process a single row using pandas apply (more efficient than iterrows).
        """
        try:
            description = row.get('order_clinical_desc', '')
            if not description:
                return None
            
            # Parse clinical description
            params = self.parse_clinical_description(description)
            self.stats.update_stats(params)
            
            if params.is_empty():
                return None
            
            # Convert Series to dict for efficiency
            row_dict = row.to_dict()
            
            # Process based on parameters
            if params.bolus:
                row_dict['volume'] = params.bolus
                return pd.DataFrame([row_dict])
                
            elif params.volume and not params.rate and not params.duration:
                row_dict['volume'] = params.volume
                return pd.DataFrame([row_dict])
                
            elif params.rate and params.duration:
                # Create time series records
                records = self._create_time_series_bulk(row, params)
                if records:
                    return pd.DataFrame(records)
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing row: {e}")
            self.stats.errors += 1
            return None


# Example usage
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--chunks", type=int, required=True)
    args = parser.parse_args()

    index = args.index
    chunks = args.chunks

    in_outs = pd.read_csv(f'/hpc/group/kamaleswaranlab/EmoryDataset/EMR_RAW/noPHI/CJSEPSIS_OUT_EO3.csv')
    
    # Chunk data into chunks number of chunks, make sure to include the entire remaining rows for the last chunk
    len_of_chunks = len(in_outs)//chunks
    if index == chunks - 1:
        in_outs = in_outs.iloc[index*len_of_chunks:]
    else:
        in_outs = in_outs.iloc[index*len_of_chunks:(index+1)*len_of_chunks]
    
    in_outs['service_ts'] = pd.to_datetime(in_outs['service_ts'])
    
    # Create processor
    processor = FluidProcessor()
    
    # Choose processing method based on your needs:
    
    print("Using vectorized processing...")
    processed_df, stats = processor.process_fluids_vectorized(
        in_outs, 
        batch_size=10000  # Larger batches are more efficient
    )
    
    
    print("Processing completed!")
    print(f"Successfully processed: {stats.processed_successfully}/{stats.total_records} records")
    print(f"Processed DataFrame shape: {processed_df.shape}")
    
    # Get summary as DataFrame if needed
    summary_df = processor.get_summary_dataframe()
    print("\nProcessing Summary:")
    print(summary_df)

    processed_df.to_csv(f'/hpc/group/kamaleswaranlab/EmoryDataset/EMR_RAW/noPHI/CJSEPSIS_IN_OUT_PROCESSED_{index}.csv', index=False)