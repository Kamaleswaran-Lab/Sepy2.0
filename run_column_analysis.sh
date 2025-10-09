#!/bin/bash
#SBATCH --job-name=column_analysis
#SBATCH --output=column_analysis_%j.out
#SBATCH --error=column_analysis_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=common

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Load any required modules (if needed)
# module load python/3.9

# Navigate to project directory
cd /hpc/dctrl/rrz4/Projects/Sepy2.0

# Activate virtual environment
echo "Activating virtual environment..."
source ENV/bin/activate

# Check Python and package versions
echo "Python version: $(python --version)"
echo "Pandas version: $(python -c 'import pandas; print(pandas.__version__)')"
echo "Matplotlib version: $(python -c 'import matplotlib; print(matplotlib.__version__)')"

# Run the column analysis
echo "Starting column summary analysis..."
python column_summary_analyzer.py

# Check if analysis completed successfully
if [ $? -eq 0 ]; then
    echo "Column analysis completed successfully!"
    echo "Results saved to: column_analysis/"
    
    # List output files
    echo "Generated files:"
    ls -la column_analysis/
    echo "Individual feature plots:"
    ls -1 column_analysis/plots/individual_features/ | wc -l
    echo "plots created"
else
    echo "Column analysis failed with exit code: $?"
fi

echo "End Time: $(date)"
echo "Job completed."
