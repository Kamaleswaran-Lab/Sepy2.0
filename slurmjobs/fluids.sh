#!/bin/bash
#SBATCH --job-name=fluids
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=./output/fluids_%A_%a.out
#SBATCH --error=./errors/fluids_%A_%a.err

python $HOME/Sepy/process_fluids.py