#!/bin/bash

# --- SLURM JOB OPTIONS ---
#SBATCH --job-name=graphrag_pipeline
#SBATCH --output=slurm_output/graphrag_job_%j.out
#SBATCH --error=slurm_output/graphrag_job_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=04:00:00
#SBATCH --partition=commons
#SBATCH --gres=gpu:1

# --- JOB SCRIPT ---

echo "Job started on $(hostname) at $(date)"
mkdir -p slurm_output

# Load CUDA module for GPU access
echo "Loading CUDA module..."
module load CUDA

# Run nvidia-smi to confirm GPU is visible
echo "Checking for GPU with nvidia-smi..."
nvidia-smi

# Activate Conda Environment
source /home/mlk15/miniforge3/etc/profile.d/conda.sh
conda activate graphrag_env

# Run the GraphRAG Initialization
#echo "Initializing a fresh GraphRAG project in ./graphRAG..."
# This command will create a default settings.yaml, .env, and prompts folder.
# We use --force to overwrite any old files and ensure a clean start.
#/home/mlk15/miniforge3/envs/graphrag_env/bin/graphrag init --root ./graphRAG --force

# Start Ollama Server in the background
echo "Starting Ollama server..."
/home/mlk15/bin/ollama serve &
OLLAMA_PID=$!
sleep 15

# Create a small data sample for development
echo "Creating data sample..."
mkdir -p ./graphRAG/input
head -n 101 ./data/cleaned_data.csv > ./graphRAG/input/dev_sample.csv

# Run the main GraphRAG pipeline
echo "Running the main GraphRAG pipeline..."
cd ./graphRAG
# The command will now automatically find and use settings.yaml
/home/mlk15/miniforge3/envs/graphrag_env/bin/graphrag index
cd ..

# 7. Shutdown Ollama Server
echo "Shutting down Ollama server..."
kill $OLLAMA_PID
wait $OLLAMA_PID 2>/dev/null

echo "Job finished at $(date)"

