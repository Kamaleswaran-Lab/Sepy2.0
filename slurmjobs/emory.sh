#!/bin/bash
#SBATCH --job-name=tomato
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=/hpc/home/ma618/Sepy/out_em/em_encounter_%A_%a.out
#SBATCH --error=/hpc/home/ma618/Sepy/out_em/em_encounter_%A_%a.err
#SBATCH --array=0-12

# This is the list of years to process; each core will take a fraction of each year
YEAR=2019

### These are variables passed into the Python Script ###
# Total num of cores assigned to job (i.e. 16)
NUM_OF_PROCESSES=$SLURM_ARRAY_TASK_COUNT


### Print basic job info to log file for quality check ###
echo The array number is- $SLURM_ARRAY_TASK_ID
echo This is the num processes - $NUM_OF_PROCESSES

## Activate the custom python environment
source /home/maror24/anaconda3/bin/activate torch

python --version

python /hpc/home/ma618/Sepy/make_dicts.py --year $YEAR --num_processes $NUM_OF_PROCESSES --processor_assignment $SLURM_ARRAY_TASK_ID