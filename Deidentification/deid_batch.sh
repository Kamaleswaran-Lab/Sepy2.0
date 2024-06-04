#!/bin/bash

#SBATCH -J deid
#SBATCH -p batch,overflow
#SBATCH -G 0
#SBATCH -t 24:0:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-7
#SBATCH --mem 72G

#SBATCH -o ./deid_%a.out
#SBATCH -e ./deid_%a.err

source /home/maror24/anaconda3/bin/deactivate
source /home/maror24/anaconda3/bin/activate rapids

export LANGUAGE=UTF-8
export LC_ALL=en_US.UTF-8
export LANG=UTF-8
export LC_CTYPE=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_COLLATE=$LANG
export LC_CTYPE=$LANG
export LC_MESSAGES=$LANG
export LC_MONETARY=$LANG
export LC_NUMERIC=$LANG
export LC_TIME=$LANG
export LC_ALL=$LANG

echo $PATH
python deidentification_grady.py --index $SLURM_ARRAY_TASK_ID
