#!/bin/bash
#SBATCH --job-name=make_dicts_2014
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/work/rrz4/Sepy2.0/slurmjobs/logs/make_dicts_2014_%j.out
#SBATCH --error=/work/rrz4/Sepy2.0/slurmjobs/logs/make_dicts_2014_%j.err

# Print basic job info to log file for quality check
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"

# Activate the virtual environment
echo "Activating virtual environment..."
source /work/rrz4/Sepy2.0/ENV/bin/activate

# Check if activation was successful
if [ $? -eq 0 ]; then
    echo "Virtual environment activated successfully"
    python --version
else
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi

# Change to the working directory
cd /work/rrz4/Sepy2.0

# Run the make_dicts.py command
echo "Starting make_dicts.py with parameters:"
echo "  Year: 2014"
echo "  Number of processes: 8"
echo "  Data config: /work/rrz4/Sepy2.0/configurations/emory_config.yaml"
echo "  Sepy config: /work/rrz4/Sepy2.0/configurations/dict_config.yaml"

python make_dicts.py \
    --year 2014 \
    --num_processes 8 \
    --data_config /work/rrz4/Sepy2.0/configurations/emory_config.yaml \
    --sepy_config /work/rrz4/Sepy2.0/configurations/dict_config.yaml

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "Job completed successfully at: $(date)"
else
    echo "ERROR: Job failed with exit code $?"
    exit 1
fi
