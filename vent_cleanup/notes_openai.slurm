#!/bin/bash
#SBATCH --job-name=notes_openai_async
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=./output/out_notes_openai_async.out
#SBATCH --error=./errors/err_notes_openai_async.err
#SBATCH --partition=compute
#SBATCH --account=pro00114885

python $HOME/Sepy2.0/vent_cleanup/cxr_notes_intubation_async.py 
