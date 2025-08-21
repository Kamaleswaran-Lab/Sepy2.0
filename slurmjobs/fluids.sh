#!/bin/bash
#SBATCH --job-name=fluids
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-15
#SBATCH --output=./output/fluids_%j_%a.out
#SBATCH --error=./errors/fluids_%j_%a.err

filepath=/labs/collab/K-lab-MODS/MODS-PHI/Fluids/CJSEPSIS_OUT_EO3.txt
savepath=/labs/collab/K-lab-MODS/MODS-PHI/Emory_Data/INs

python $HOME/Sepy2.0/data_preprocess/process_fluids.py --chunks 16 --index $SLURM_ARRAY_TASK_ID --filepath $filepath --savepath $savepath