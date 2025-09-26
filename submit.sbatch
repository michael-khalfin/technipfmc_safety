#!/bin/bash

#SBATCH --job-name=fmcsafety_eda
#SBATCH --output=slurm_output/eda_job_%j.out   # Creates a folder for output files
#SBATCH --error=slurm_output/eda_job_%j.err    # Creates a folder for error files
#SBATCH --nodes=1
#SBATCH --ntasks=1                             
#SBATCH --cpus-per-task=1                      
#SBATCH --time=00:15:00                        # A reasonable 15-minute runtime
#SBATCH --mem-per-cpu=4G                       # A safe 4GB memory request for pandas
#SBATCH --partition=commons                       

# Create a directory for slurm output files if it doesn't exist
mkdir -p slurm_output

# Activate the Python virtual environment
echo "Activating Python virtual environment..."
source .venv/bin/activate

# Run EDA script
echo "Running the EDA script from eda/main.py..."
python3 eda/main.py

echo "Job finished."