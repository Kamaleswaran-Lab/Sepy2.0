#!/bin/bash
#SBATCH --job-name=mimicstats
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=./output/mimicstats_%A_%a.out
#SBATCH --error=./errors/mimicstats_%A_%a.err

ROOT="/hpc/group/kamaleswaranlab/mimic_iv/sepy_output/mimic-supertables/"
STATS_DF_OUTPUT_PATH="/hpc/group/kamaleswaranlab/mimic_iv/stats/"

mkdir -p $STATS_DF_OUTPUT_PATH
mkdir -p ./output
mkdir -p ./errors

mkdir -p $STATS_DF_OUTPUT_PATH/"histograms"/

# Run Python script with proper line continuations
python $HOME/Sepy2.0/data_quality_check/generate_db_stats.py \
    --data_path $ROOT/Supertables \
    --stats_df_output_path $STATS_DF_OUTPUT_PATH/stats.csv \
    --columns_config_path $HOME/Sepy2.0/configurations/columns_config_sepy.csv \
    --parquet_output_path $STATS_DF_OUTPUT_PATH/stats.parquet \
    --histogram \
    --histogram_output_path $STATS_DF_OUTPUT_PATH/histograms/

echo "Done"