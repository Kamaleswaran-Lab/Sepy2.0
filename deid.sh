#!/bin/bash

#SBATCH -J deidH
#SBATCH -p batch,overflow
#SBATCH -t 24:0:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-5

#SBATCH -o ./OutputErrorFiles/deid_%a.out
#SBATCH -e ./OutputErrorFiles/deid_%a.err

#SBATCH --mem 32G

years=(2014 2015 2016 2017 2018 2019 2020 2021 2022)

source /home/maror24/anaconda3/bin/deactivate
source /home/maror24/anaconda3/bin/activate rapids

echo "This task is : $SLURM_ARRAY_TASK_ID"
echo "Year : ${years[$SLURM_ARRAY_TASK_ID]}"

python deidentify_supertables.py --year ${years[$SLURM_ARRAY_TASK_ID]}

echo "Done"
