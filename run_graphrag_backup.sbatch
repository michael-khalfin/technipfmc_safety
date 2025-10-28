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

# 1. Load system modules
module load foss/2023b
module load Python/3.11.5

# Load CUDA module for GPU access
echo "Loading CUDA module..."
module load CUDA/12.4.0
module load Miniforge3/24.11.3-0

# 2. Initialize Conda and activate your environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate graphrag_env
export PATH=$CONDA_PREFIX/bin:$PATH

# Set OLLAMA GPU Environment Variables
export LD_LIBRARY_PATH=/opt/apps/software/CUDA/12.4.0/lib64:$LD_LIBRARY_PATH
export OLLAMA_LLM_LIBRARY=cuda
export CUDA_VISIBLE_DEVICES=0

# Run nvidia-smi to confirm GPU is visible
echo "Checking for GPU with nvidia-smi..."
nvidia-smi

# Kill any existing Ollama process using port 11434
echo "Killing any previous Ollama process on port 11434..."
PORT=11434  
PID_ON_PORT=$(lsof -ti tcp:$PORT)
if [ ! -z "$PID_ON_PORT" ]; then
  echo "Found process on port $PORT: $PID_ON_PORT. Killing it..."
  kill -9 $PID_ON_PORT
fi

# Run the GraphRAG Initialization
#echo "Initializing a fresh GraphRAG project in ./graphRAG..."
# This command will create a default settings.yaml, .env, and prompts folder.
# We use --force to overwrite any old files and ensure a clean start.
graphrag init --root ./graphRAG --force

# Start Ollama Server in the background
echo "Starting Ollama server..."
~/bin/ollama serve &
OLLAMA_PID=$!
sleep 15

echo "Checking Ollama system info..."

# Create a small data sample for development
echo "Creating data sample..."
mkdir -p ./graphRAG/input
head -n 1001 ./data/cleaned_data.csv > ./graphRAG/input/dev_sample.csv

# Run the main GraphRAG pipeline
echo "Running the main GraphRAG pipeline..."
cd ./graphRAG
graphrag index
cd .. 

# 7. Shutdown Ollama Server
echo "Shutting down Ollama server..."
kill $OLLAMA_PID
wait $OLLAMA_PID 2>/dev/null

echo "Job finished at $(date)"

