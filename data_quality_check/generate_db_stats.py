import pandas as pd
import numpy as np
from pathlib import Path
import os 
import duckdb
import sys
import argparse

sys.path.append("../")

import columns_map as cm
import stats_generator as sg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type = str, required = True)
    parser.add_argument("--stats_df_output_path", type = str, required = True)
    parser.add_argument("--columns_config_path", type = str, required = True)
    parser.add_argument("--parquet_output_path", type = str, required = True)
    parser.add_argument("--histogram", action='store_true', default=False)
    parser.add_argument("--histogram_output_path", type = str, required = False)
    args = parser.parse_args()
    
    files = os.listdir(args.data_path)

    if args.histogram and args.histogram_output_path is None:
        raise ValueError("Histogram output path is required if histogram is True")

    stats_df, hist_data = sg.calculate_cohort_statistics_duckdb(
        files_dir = args.data_path,
        selected_files = files,
        config_file_path = args.columns_config_path,
        plot_histograms = args.histogram,
        histogram_dir = args.histogram_output_path,
        save_parquet_path = args.parquet_output_path
    )
    
    stats_df.to_csv(args.stats_df_output_path)

if __name__ == "__main__":
    main()