#!/bin/bash
#SBATCH --job-name=fluids
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-12
#SBATCH --output=./output/fluids_%j_%a.out
#SBATCH --error=./errors/fluids_%j_%a.err

filepath=
savepath=/labs/collab/K-lab-MODS/MODS-PHI/Emory_Data/INs

python $HOME/Sepy/process_fluids.py --chunks 10 --index $SLURM_ARRAY_TASK_ID