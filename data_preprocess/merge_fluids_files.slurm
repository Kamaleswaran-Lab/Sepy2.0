#!/bin/bash
#SBATCH --job-name=merge_fluids
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=./output/out_merge_fluids.out
#SBATCH --error=./errors/err_merge_fluids.err
#SBATCH --partition=compute
#SBATCH --account=pro00114885

python $HOME/Sepy2.0/data_preprocess/merge_fluids_files.py 
