#!/bin/bash
#SBATCH --job-name=combine_chunks
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=./output/combine_chunks_%j_%a.out
#SBATCH --error=./errors/combine_chunks_%j_%a.err

python $HOME/Sepy/combine_chunks.py --root /hpc/group/kamaleswaranlab/EmoryDataset/EMR_RAW/noPHI --glob_template CJSEPSIS_IN_OUT_PROCESSED_*.csv

