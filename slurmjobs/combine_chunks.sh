#!/bin/bash
#SBATCH --job-name=combine_chunks
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=./output/combine_chunks_fluids.out
#SBATCH --error=./errors/combine_chunks_fluids.err

python $HOME/Sepy2.0/data_preprocess/combine_chunks.py --root /labs/collab/K-lab-MODS/MODS-PHI/Emory_Data/INs --glob_template CJSEPSIS_IN_OUT_PROCESSED_*.csv

