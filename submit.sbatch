#!/bin/bash
#SBATCH --job-name=python_test
#SBATCH --output=python_test_%j.out
#SBATCH --error=python_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=02:00:00
#SBATCH --partition=nots

# Load Python

# Run your script
python my_script.py
