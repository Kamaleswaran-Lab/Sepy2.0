#!/bin/bash
#SBATCH --job-name=grstats
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=./output/grstats_%A_%a.out
#SBATCH --error=./errors/grstats_%A_%a.err
#SBATCH --array=0-8

ROOT="/hpc/group/kamaleswaranlab/GradyDataset/sepy_processed/grady_supertables"
STATS_DF_OUTPUT_PATH="/hpc/group/kamaleswaranlab/GradyDataset/stats/"

mkdir -p $STATS_DF_OUTPUT_PATH
mkdir -p ./output
mkdir -p ./errors

YEARS=(2014 2015 2016 2017 2018 2019 2020 2021 2022)

mkdir -p $STATS_DF_OUTPUT_PATH/"histograms_${YEARS[$SLURM_ARRAY_TASK_ID]}"/

# Run Python script with proper line continuations
python $HOME/Sepy2.0/data_quality_check/generate_db_stats.py \
    --data_path $ROOT/${YEARS[$SLURM_ARRAY_TASK_ID]}/Supertables \
    --stats_df_output_path $STATS_DF_OUTPUT_PATH/stats_${YEARS[$SLURM_ARRAY_TASK_ID]}.csv \
    --columns_config_path $HOME/Sepy2.0/configurations/columns_config_sepy.csv \
    --parquet_output_path $STATS_DF_OUTPUT_PATH/stats_${YEARS[$SLURM_ARRAY_TASK_ID]}.parquet \
    --histogram \
    --histogram_output_path $STATS_DF_OUTPUT_PATH/histograms_${YEARS[$SLURM_ARRAY_TASK_ID]}/

echo "Done"