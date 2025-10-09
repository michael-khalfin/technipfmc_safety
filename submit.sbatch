#!/bin/bash

#SBATCH --job-name=fmcsafety_eda
#SBATCH --output=slurm_output/eda_job_%j.out   # Creates a folder for output files
#SBATCH --error=slurm_output/eda_job_%j.err    # Creates a folder for error files
#SBATCH --nodes=1
#SBATCH --ntasks=1                             
#SBATCH --cpus-per-task=4  
#SBATCH --mem=16G                       # A safe 4GB memory request for pandas                   
#SBATCH --partition=commons                       
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

# Create a directory for slurm output files if it doesn't exist
mkdir -p slurm_output

# Load environment
chmod +x ner_init.bash
./ner_init.bash

# Run your Python job
echo "Running NER"
python3 eda/ner.py

echo "Job finished."