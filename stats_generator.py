import pandas as pd
import numpy as np
import os
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from functools import partial
from utils import convert_dtypes_supertable
import duckdb
from typing import Union, Optional, List, Tuple, Dict

def compute_stats_for_supertable(file_path):
    """
    Computes statistics for a single supertable file.
    
    Args:
        file_path (Path): Path to the supertable pickle file
        
    Returns:
        dict: Dictionary containing statistics for each numeric column
    """
    try:
        # Extract patient_id from filename
        patient_id = str(file_path.stem).split('_')[0]
        
        # Load the dataframe
        df = pd.read_pickle(file_path)
        
        # Convert dtypes
        df = convert_dtypes_supertable(df)
        
        # Initialize results dictionary
        stats = {'patient_id': patient_id}
        
        # Separate columns by type (excluding specific columns)
        exclude_cols = ['encounter_id', 'cxr_timing', 'cxr_timing_approx_flag', 'cxr_timing_ffill']
        
        # Get numeric and string columns
        numeric_cols = []
        string_cols = []
        for col in df.columns:
            if col not in exclude_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_cols.append(col)
                elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                    string_cols.append(col)
        
        # Calculate statistics for numeric columns
        for col in numeric_cols:
            # Count NaN values
            nan_count = df[col].isna().sum()
            # Get non-NaN values for other statistics
            values = df[col].dropna()
            
            # Skip if all values are NaN
            if len(values) == 0:
                stats[f"{col}_min"] = np.nan
                stats[f"{col}_max"] = np.nan
                stats[f"{col}_median"] = np.nan
                stats[f"{col}_mean"] = np.nan
                stats[f"{col}_sum"] = np.nan
                stats[f"{col}_std"] = np.nan
                stats[f"{col}_nan_count"] = nan_count
                continue
                
            # Calculate statistics
            stats[f"{col}_min"] = values.min()
            stats[f"{col}_max"] = values.max()
            stats[f"{col}_median"] = values.median()
            stats[f"{col}_mean"] = values.mean()
            stats[f"{col}_sum"] = values.sum()
            stats[f"{col}_std"] = values.std()
            stats[f"{col}_nan_count"] = nan_count
            
        # Calculate statistics for string/categorical columns
        for col in string_cols:
            # Count NaN/null values
            nan_count = df[col].isna().sum()
            stats[f"{col}_nan_count"] = nan_count
            
            # Get value counts (excluding NaN)
            value_counts = df[col].value_counts(dropna=True)
            
            # Store unique value counts
            stats[f"{col}_unique_count"] = len(value_counts)
            
            # Store values as a list of strings
            stats[f"{col}_values"] = value_counts.index.tolist()
            stats[f"{col}_counts"] = value_counts.values.tolist()
            
        return stats
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return {'patient_id': patient_id, 'error': str(e)}

def generate_patient_stats(input_dir=None, files=None, output_file=None, num_workers=None, chunk_size=10) -> pd.DataFrame:
    """
    Generates patient-level statistics from individual supertable files using multiprocessing.
    
    This function processes each patient's supertable file separately to compute statistics
    for numeric and categorical columns within that patient's data. The output is a DataFrame
    where each row represents one patient and columns contain various statistics (min, max, 
    mean, etc.) for each variable in the supertables.

    Args:
        input_dir (str or Path, optional): Directory containing processed supertable files 
            (files matching pattern '*_processed.pkl'). Either this or 'files' must be provided.
        files (list of str or Path, optional): Explicit list of file paths to process.
            Either this or 'input_dir' must be provided.
        output_file (str or Path, optional): Path to save the output DataFrame. Supports
            .csv and .pickle formats. If no extension provided, defaults to .pickle.
        num_workers (int, optional): Number of worker processes for multiprocessing.
            Defaults to CPU count - 1.
        chunk_size (int, optional): Number of files processed per worker batch.
            Defaults to 10. Larger values may improve efficiency for many small files.

    Returns:
        pd.DataFrame: DataFrame indexed by patient_id with columns containing statistics.
            Column naming convention: '{original_column}_{statistic}' (e.g., 'heart_rate_mean')
            
    Raises:
        ValueError: If neither input_dir nor files is provided, or if no valid files found.
        
    Example:
        >>> # Process all files in directory
        >>> stats_df = generate_patient_stats(
        ...     input_dir='/path/to/processed_supertables/',
        ...     output_file='/path/to/patient_stats.csv',
        ...     num_workers=8
        ... )
        >>> print(f"Generated stats for {len(stats_df)} patients")
        
        >>> # Process specific files
        >>> file_list = ['patient1_processed.pkl', 'patient2_processed.pkl']
        >>> stats_df = generate_patient_stats(files=file_list)
    """
    if files is not None:
        files = [Path(f) for f in files]
    elif input_dir is not None:
        input_dir = Path(input_dir)
        files = list(input_dir.glob("*_processed.pkl"))
        #if len(files) == 0:
        #    files = list(input_dir.glob("*.csv"))
        #if len(files) == 0:
        #    files = list(input_dir.glob("*.pickle"))
        #if len(files) == 0:
        #    raise ValueError("No supertable files found in input directory")
    else:
        raise ValueError("Either input_dir or files must be provided.")

    print(f"Found {len(files)} supertable files to process")

    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    print(f"Processing with {num_workers} workers, chunk size {chunk_size}")
    results = []

    with mp.Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap(compute_stats_for_supertable, files, chunksize=chunk_size),
                           total=len(files)):
            if result:
                results.append(result)

    stats_df = pd.DataFrame(results)

    if 'patient_id' in stats_df.columns:
        stats_df.set_index('patient_id', inplace=True)

    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if output_file.suffix == '.csv':
            stats_df.to_csv(output_file)
        elif output_file.suffix == '.pickle':
            stats_df.to_pickle(output_file)
        else:
            stats_df.to_pickle(output_file.with_suffix('.pickle'))

        print(f"Saved statistics to {output_file}")

    return stats_df


def calculate_cohort_statistics(files_dir, selected_columns, selected_files=None, quantiles=[0.25, 0.5, 0.75]):
    """
    Calculates population-level statistics across the entire patient cohort.
    
    This function loads multiple patient supertable files, concatenates them into a single
    large dataset, and computes cohort-wide statistics for specified columns. Unlike
    generate_patient_stats() which analyzes each patient separately, this function provides
    population-level insights by pooling all patient data together.

    Args:
        files_dir (str or Path): Directory containing .pkl supertable files to analyze.
        selected_columns (list): List of column names to include in the statistical analysis.
            Only these columns will be processed and included in the output.
        selected_files (list, optional): List of specific filenames to process within files_dir.
            If None, all .pkl files in the directory will be processed. Defaults to None.
        quantiles (list, optional): List of quantile values to calculate for numeric columns.
            Values should be between 0 and 1. Defaults to [0.25, 0.5, 0.75] (quartiles).
        
    Returns:
        pd.DataFrame: DataFrame where each row represents one column from selected_columns
            and columns contain various statistics:
            - For numeric columns: count, nan_count, mean, median, std, min, max, range,
              quantiles (q25, q50, q75), and iqr
            - For categorical columns: count, nan_count, unique_count, most_frequent,
              most_frequent_count, type
              
    Raises:
        ValueError: If no .pkl files found in files_dir or no valid dataframes could be loaded.
        
    Example:
        >>> # Analyze specific columns across all patients
        >>> columns_of_interest = ['heart_rate', 'blood_pressure', 'age', 'gender']
        >>> cohort_stats = calculate_cohort_statistics(
        ...     files_dir='/path/to/supertables/',
        ...     selected_columns=columns_of_interest,
        ...     quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]
        ... )
        >>> print(f"Heart rate mean: {cohort_stats.loc['heart_rate', 'mean']:.2f}")
        
        >>> # Analyze only test set patients
        >>> test_files = ['patient1.pkl', 'patient2.pkl', 'patient3.pkl']
        >>> test_stats = calculate_cohort_statistics(
        ...     files_dir='/path/to/supertables/',
        ...     selected_columns=columns_of_interest,
        ...     selected_files=test_files
        ... )
    """
    files_dir = Path(files_dir)
    if selected_files is None:
        pkl_files = list(files_dir.glob("*.pkl"))
    else:
        pkl_files = [files_dir / file for file in selected_files]
    
    if len(pkl_files) == 0:
        raise ValueError(f"No .pkl files found in {files_dir}")
    
    print(f"Found {len(pkl_files)} .pkl files to process")
    
    # Load and concatenate all dataframes
    all_dfs = []
    for file_path in tqdm(pkl_files, desc="Loading files"):
        try:
            df = pd.read_pickle(file_path)
            df = df.reset_index(drop=True)

            # Filter to selected columns (keep only those that exist)
            available_cols = [col for col in selected_columns if col in df.columns]
            if available_cols:
                df_filtered = df[available_cols]
                # Remove duplicate columns before adding to list
                df_filtered = df_filtered.loc[:, ~df_filtered.columns.duplicated()]
                all_dfs.append(df_filtered)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            continue
    
    if not all_dfs:
        raise ValueError("No valid dataframes could be loaded")
    
    # Concatenate all dataframes
    cohort_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined dataset shape: {cohort_df.shape}")
    
    # Calculate statistics for each column
    stats_dict = {}

    for col in selected_columns:
        if col not in cohort_df.columns:
            print(f"Warning: Column '{col}' not found in any files")
            continue
            
        series = cohort_df[col]
        
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(series):
            # Drop NaN values for calculations
            series_clean = series.dropna()
            
            if len(series_clean) == 0:
                print(f"Warning: Column '{col}' has no non-NaN values")
                continue
            
            # Handle boolean columns by converting to int for calculations
            if pd.api.types.is_bool_dtype(series_clean):
                series_clean = series_clean.astype(int)
            
            # Calculate basic statistics
            stats_dict[col] = {
                'count': len(series_clean),
                'nan_count': series.isna().sum(),
                'mean': series_clean.mean(),
                'median': series_clean.median(),
                'std': series_clean.std(),
                'min': series_clean.min(),
                'max': series_clean.max(),
                'range': series_clean.max() - series_clean.min()
            }
            
            # Calculate quantiles
            for q in quantiles:
                stats_dict[col][f'q{int(q*100)}'] = series_clean.quantile(q)
            
            # Calculate IQR (if 25th and 75th percentiles are available)
            if 0.25 in quantiles and 0.75 in quantiles:
                q25 = series_clean.quantile(0.25)
                q75 = series_clean.quantile(0.75)
                stats_dict[col]['iqr'] = q75 - q25
            
        else:
            # For categorical/string columns
            value_counts = series.value_counts(dropna=True)
            stats_dict[col] = {
                'count': len(series.dropna()),
                'nan_count': series.isna().sum(),
                'unique_count': len(value_counts),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'type': 'categorical'
            }
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(stats_dict).T
    
    return stats_df



def calculate_cohort_statistics_duckdb(
    files_dir: Union[str, Path], 
    config_file_path: Union[str, Path],
    selected_files: Optional[List] = None, 
    quantiles: Optional[List] = [0.01, 0.25, 0.5, 0.75, 0.99],
    plot_histograms: bool = False, 
    histogram_dir: Optional[Union[str, Path]] = None, 
    num_bins: Optional[int] = 50, 
    save_as_pdf: Optional[bool] = False, 
    figsize: Optional[Tuple] = (10, 6), 
    dpi: Optional[int] = 100,
    save_parquet_path: Optional[Union[str, Path]] = None
    ) -> Tuple[pd.DataFrame, Dict]:
    """
    Calculates population-level statistics across the entire patient cohort using DuckDB.
    
    Loads files incrementally into DuckDB instead of concatenating all files in pandas memory at once. 
    Optionally generates histograms for numeric columns in the same pass.

    Args:
        files_dir (str or Path): Directory containing .pkl supertable files to analyze.
        config_file_path (str or Path): Path to CSV config file with columns: column_name, dtype, 
            include_in_stats. Will use this to determine which columns to analyze and 
            their types.
        selected_files (list, optional): List of specific filenames to process within files_dir.
            If None, all .pkl files in the directory will be processed. Defaults to None.
        quantiles (list, optional): List of quantile values to calculate for numeric columns.
            Values should be between 0 and 1. Defaults to [0.01, 0.25, 0.5, 0.75, 0.99].
        plot_histograms (bool, optional): If True, generates histogram plots for all numeric
            columns. Defaults to False.
        histogram_dir (str or Path, optional): Directory where histogram files will be saved
            (only used if plot_histograms=True). Defaults to 'histograms/'.
        num_bins (int, optional): Number of bins for histograms. Defaults to 50.
        save_as_pdf (bool, optional): If True, saves all histograms in a single PDF file.
            If False, saves each as a separate PNG. Defaults to False.
        figsize (tuple, optional): Figure size as (width, height) in inches. Defaults to (10, 6).
        dpi (int, optional): Resolution for saved images. Defaults to 100.
        save_parquet_path (str or Path, optional): If provided, saves the DuckDB table as a 
            Parquet file for later querying. Includes a 'csn' column extracted from filenames.
            Defaults to None.
        
    Returns:
        tuple: Returns (stats_df, histogram_data) where stats_df is a DataFrame with statistics
            and histogram_data is a dict (empty if plot_histograms=False).
            
            DataFrame columns contain various statistics:
            - For numeric columns: count, nan_count, mean, median, std, min, max, range,
              quantiles (q01, q25, q50, q75, q99), and iqr
            - For categorical columns: count, nan_count, unique_count, most_frequent,
              most_frequent_count, type
              
    Example:
        >>> # Use config file to automatically determine columns and types
        >>> stats_df, hist_data = calculate_cohort_statistics_duckdb(
        ...     files_dir='/path/to/supertables/',
        ...     config_file_path='configurations/columns_config_sepy.csv',
        ...     plot_histograms=True
        ... )
        
        >>> # Save the full dataset as Parquet for later querying
        >>> stats_df, hist_data = calculate_cohort_statistics_duckdb(
        ...     files_dir='/path/to/supertables/',
        ...     config_file_path='configurations/columns_config_sepy.csv',
        ...     save_parquet_path='output/cohort_data.parquet'
        ... )
        >>> # Later, query the parquet file:
        >>> import duckdb
        >>> result = duckdb.query("SELECT * FROM 'output/cohort_data.parquet' WHERE csn = '12345'").df()
    """
    conn = duckdb.connect()  # In-memory database
    
    # Load column configuration if provided
    column_types = {}  # Maps column_name -> DuckDB type
    try:
        config_file_path = Path(config_file_path)
        print(f"Loading column configuration from {config_file_path}")
        
        config_df = pd.read_csv(config_file_path)
        
        # Filter to columns where include_in_stats is TRUE
        config_df = config_df[config_df['include_in_stats'].astype(str).str.upper() == 'TRUE']
        
        # Get selected columns from config
        selected_columns = config_df['column_name'].tolist()
        
        # Map pandas dtype to DuckDB type
        dtype_map = {
            'int': 'BIGINT',
            'float': 'DOUBLE',
            'string': 'VARCHAR'
        }
        
        # Build column_types dictionary
        for _, row in config_df.iterrows():
            col_name = row['column_name']
            dtype = row['dtype'].lower()
            duckdb_type = dtype_map.get(dtype, 'VARCHAR')
            column_types[col_name] = duckdb_type
        
        print(f"Loaded {len(selected_columns)} columns from config (where include_in_stats=TRUE)")
    
    except Exception as e:
        print(f"Error reading config file: {e}")

    files_dir = Path(files_dir)
    if selected_files is None:
        pkl_files = list(files_dir.glob("*.pkl"))
    else:
        pkl_files = [files_dir / file for file in selected_files]
    
    if len(pkl_files) == 0:
        raise ValueError(f"No .pkl files found in {files_dir}")
    
    print(f"Found {len(pkl_files)} .pkl files to process")
    
    # Create a temporary table by appending data from each file
    first_file = True
    table_created = False
    
    for file_path in tqdm(pkl_files, desc="Loading files into DuckDB"):
        try:
            df = pd.read_pickle(file_path)
            df = df.reset_index(drop=True)
            
            # Extract CSN from filename (everything before first dot)
            csn = file_path.stem.split('.')[0]
            
            # Filter to selected columns (keep only those that exist)
            if selected_columns is None and first_file:
                selected_columns = df.columns.tolist()

            available_cols = [col for col in selected_columns if col in df.columns]
            if available_cols:
                df_filtered = df[available_cols]
                # Remove duplicate columns before adding to list
                df_filtered = df_filtered.loc[:, ~df_filtered.columns.duplicated()]
                
                # Add CSN column
                df_filtered['csn'] = csn
                
                # Cast columns to appropriate types if column_types dict exists
                if column_types:
                    for col in df_filtered.columns:
                        if col in column_types:
                            target_type = column_types[col]
                            try:
                                if target_type == 'BIGINT':
                                    df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
                                    # Replace inf with NaN
                                    df_filtered[col] = df_filtered[col].replace([np.inf, -np.inf], np.nan)
                                    df_filtered[col] = df_filtered[col].astype('Int64')
                                elif target_type == 'DOUBLE':
                                    df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
                                    # Replace inf with NaN
                                    df_filtered[col] = df_filtered[col].replace([np.inf, -np.inf], np.nan)
                                elif target_type == 'VARCHAR':
                                    df_filtered[col] = df_filtered[col].astype(str)
                            except Exception as cast_error:
                                print(f"Warning: Could not cast column '{col}' to {target_type}: {cast_error}")
                
                # Register this DataFrame and insert into persistent table
                if first_file:
                    if column_types:
                        # Create table with explicit schema
                        # Build CREATE TABLE statement with explicit types
                        col_defs = []
                        for col in df_filtered.columns:
                            if col == 'csn':
                                # CSN column should be VARCHAR
                                col_defs.append('"csn" VARCHAR')
                            else:
                                duckdb_type = column_types.get(col, 'VARCHAR')
                                # Escape column names with double quotes to handle special characters
                                col_defs.append(f'"{col}" {duckdb_type}')
                        
                        create_stmt = f"CREATE TABLE cohort_data ({', '.join(col_defs)})"
                        conn.execute(create_stmt)
                        print(f"Created table with explicit schema for {len(col_defs)} columns (including csn)")
                        table_created = True
                    else:
                        # Create table from first DataFrame (infer types)
                        conn.execute("CREATE TABLE cohort_data AS SELECT * FROM df_filtered")
                        table_created = True
                    
                    first_file = False
                
                # Insert data (whether first file or not, we need to insert after table creation)
                if table_created:
                    conn.execute("INSERT INTO cohort_data SELECT * FROM df_filtered")
                    
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    if not table_created:
        raise ValueError("No valid dataframes could be loaded")
    
    # Get table info to check combined size
    table_info = conn.execute("SELECT COUNT(*) as row_count FROM cohort_data").df()
    print(f"Combined dataset rows: {table_info['row_count'].iloc[0]}")
    
    # Calculate statistics for each column
    stats_dict = {}
    
    for col in selected_columns:
        # Check if column exists in the table
        try:
            col_check = conn.execute(f"SELECT '{col}' FROM cohort_data LIMIT 1").fetchall()
        except:
            print(f"Warning: Column '{col}' not found in any files")
            continue
        
        # Check if column is numeric
        if column_types and col in column_types:
            # Use explicit type from config
            is_numeric = column_types[col] in ['BIGINT', 'DOUBLE', 'INT', 'FLOAT', 'INTEGER']
        else:
            # Fallback: check using DuckDB's type system
            # Get a sample to check if column is numeric or categorical
            sample = conn.execute(f"SELECT \"{col}\" FROM cohort_data WHERE \"{col}\" IS NOT NULL LIMIT 1").df()
            
            if len(sample) == 0:
                print(f"Warning: Column '{col}' has no non-NaN values")
                continue
            
            col_type = conn.execute(f"""
                SELECT typeof(\"{col}\") as col_type 
                FROM cohort_data 
                WHERE \"{col}\" IS NOT NULL 
                LIMIT 1
            """).fetchdf()
            
            type_name = col_type['col_type'].iloc[0].upper()
            is_numeric = any(t in type_name for t in ['INT', 'DOUBLE', 'FLOAT', 'DECIMAL', 'NUMERIC', 'BIGINT', 'SMALLINT', 'TINYINT', 'HUGEINT', 'BOOLEAN'])
        
        # Check for non-null values (only needed if we didn't already check above)
        if column_types and col in column_types:
            sample_check = conn.execute(f"SELECT COUNT(\"{col}\") as cnt FROM cohort_data WHERE \"{col}\" IS NOT NULL").fetchone()
            if sample_check[0] == 0:
                print(f"Warning: Column '{col}' has no non-NaN values")
                continue
        
        if is_numeric:
            try:
                # Build quantile list for query
                quantile_selects = ",\n                ".join([
                    f"QUANTILE_CONT(\"{col}\", {q}) as q{int(q*100)}" 
                    for q in quantiles
                ])
                
                query = f"""
                SELECT 
                    COUNT(\"{col}\") as count,
                    COUNT(*) - COUNT(\"{col}\") as nan_count,
                    AVG(\"{col}\") as mean,
                    MEDIAN(\"{col}\") as median,
                    STDDEV(\"{col}\") as std,
                    MIN(\"{col}\") as min,
                    MAX(\"{col}\") as max,
                    MAX(\"{col}\") - MIN(\"{col}\") as range,
                    {quantile_selects}
                FROM cohort_data
                """
                
                result = conn.execute(query).df()
                
                # Check if we got valid results
                if result['count'].iloc[0] == 0:
                    print(f"Warning: Column '{col}' has no non-NaN values")
                    continue
                
                stats_dict[col] = result.iloc[0].to_dict()
                
                # Add IQR if applicable
                if 0.25 in quantiles and 0.75 in quantiles:
                    stats_dict[col]['iqr'] = stats_dict[col]['q75'] - stats_dict[col]['q25']
                    
            except Exception as e:
                print(f"Error computing numeric stats for '{col}': {str(e)}")
                continue
        else:
            # For categorical/string columns
            try:
                # Get basic counts
                count_query = f"""
                SELECT 
                    COUNT(\"{col}\") as count,
                    COUNT(*) - COUNT(\"{col}\") as nan_count
                FROM cohort_data
                """
                count_result = conn.execute(count_query).df()
                
                # Get value counts for most frequent
                value_counts_query = f"""
                SELECT 
                    \"{col}\",
                    COUNT(*) as freq
                FROM cohort_data
                WHERE \"{col}\" IS NOT NULL
                GROUP BY \"{col}\"
                ORDER BY freq DESC
                """
                value_counts = conn.execute(value_counts_query).df()
                
                stats_dict[col] = {
                    'count': int(count_result['count'].iloc[0]),
                    'nan_count': int(count_result['nan_count'].iloc[0]),
                    'unique_count': len(value_counts),
                    'most_frequent': value_counts[col].iloc[0] if len(value_counts) > 0 else None,
                    'most_frequent_count': int(value_counts['freq'].iloc[0]) if len(value_counts) > 0 else 0,
                    'type': 'categorical'
                }
                
            except Exception as e:
                print(f"Error computing categorical stats for '{col}': {str(e)}")
                continue
    
    # Generate histograms if requested
    histogram_data = {}
    if plot_histograms:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        
        if histogram_dir is None:
            histogram_dir = 'histograms/'
        
        histogram_dir = Path(histogram_dir)
        histogram_dir.mkdir(parents=True, exist_ok=True)
        
        # Get list of numeric columns that have stats
        numeric_columns = [col for col in stats_dict.keys() 
                          if 'mean' in stats_dict[col]]  # Only numeric columns have 'mean'
        
        if len(numeric_columns) == 0:
            print("No numeric columns found to plot")
        else:
            print(f"Generating histograms for {len(numeric_columns)} numeric columns")
            
            # Set up PDF if needed
            if save_as_pdf:
                pdf_path = histogram_dir / 'cohort_histograms.pdf'
                pdf = PdfPages(pdf_path)
            
            # Generate histogram for each numeric column
            for col in tqdm(numeric_columns, desc="Generating histograms"):
                try:
                    # Get min/max from already calculated stats
                    min_val = stats_dict[col]['min']
                    max_val = stats_dict[col]['max']
                    total_count = stats_dict[col]['count']
                    
                    if min_val == max_val:
                        print(f"Skipping '{col}': all values are identical ({min_val})")
                        continue
                    
                    # Calculate bin width
                    bin_width = (max_val - min_val) / num_bins
                    
                    # Calculate histogram bins manually (WIDTH_BUCKET not available in all DuckDB versions)
                    hist_query = f"""
                    SELECT 
                        LEAST(
                            CAST(FLOOR((\"{col}\" - {min_val}) / {bin_width}) AS INTEGER) + 1,
                            {num_bins}
                        ) as bin,
                        COUNT(*) as count
                    FROM cohort_data
                    WHERE \"{col}\" IS NOT NULL
                    GROUP BY bin
                    ORDER BY bin
                    """
                    hist_data = conn.execute(hist_query).df()
                    bins = hist_data['bin'].values
                    counts = hist_data['count'].values
                    bin_centers = min_val + (bins - 0.5) * bin_width
                    
                    # Store histogram data
                    histogram_data[col] = {
                        'bins': bins.tolist(),
                        'counts': counts.tolist(),
                        'bin_centers': bin_centers.tolist(),
                        'min': float(min_val),
                        'max': float(max_val)
                    }
                    
                    # Create plot
                    fig, ax = plt.subplots(figsize=figsize)
                    ax.bar(bin_centers, counts, width=bin_width * 0.9, edgecolor='black', alpha=0.7)
                    ax.set_xlabel(col, fontsize=12)
                    ax.set_ylabel('Frequency', fontsize=12)
                    ax.set_title(f'Histogram of {col}\n(n={total_count:,})', fontsize=14)
                    ax.grid(axis='y', alpha=0.3)
                    
                    # Add basic statistics to plot (from already calculated stats)
                    stats_text = f'Mean: {stats_dict[col]["mean"]:.2f}\nMedian: {stats_dict[col]["median"]:.2f}\nStd: {stats_dict[col]["std"]:.2f}'
                    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
                           fontsize=10, verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    
                    plt.tight_layout()
                    
                    # Save plot
                    if save_as_pdf:
                        pdf.savefig(fig, dpi=dpi)
                    else:
                        png_path = histogram_dir / f'{col}_histogram.png'
                        plt.savefig(png_path, dpi=dpi, bbox_inches='tight')
                    
                    plt.close(fig)
                    
                except Exception as e:
                    print(f"Error generating histogram for '{col}': {str(e)}")
                    continue
            
            # Close PDF if used
            if save_as_pdf:
                pdf.close()
                print(f"Saved all histograms to {pdf_path}")
            else:
                print(f"Saved {len(histogram_data)} histograms to {histogram_dir}")
    
    # Save to Parquet if requested
    if save_parquet_path is not None:
        save_parquet_path = Path(save_parquet_path)
        save_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving DuckDB table to Parquet: {save_parquet_path}")
        conn.execute(f"COPY cohort_data TO '{save_parquet_path}' (FORMAT PARQUET)")
        print(f"Successfully saved {save_parquet_path}")
    
    conn.close()
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(stats_dict).T
    

    return stats_df, histogram_data