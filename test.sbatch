#!/bin/bash

#SBATCH --job-name=fmcsafety_eda
#SBATCH --output=slurm_output/test_job_%j.out   # Creates a folder for output files
#SBATCH --error=slurm_output/test_job_%j.err    # Creates a folder for error files
#SBATCH --nodes=1
#SBATCH --ntasks=1                             
#SBATCH --cpus-per-task=1                      
#SBATCH --time=00:15:00                        # A reasonable 15-minute runtime
#SBATCH --mem-per-cpu=4G                       # A safe 4GB memory request for pandas
#SBATCH --partition=commons                       

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

# Start Ollama Server in the background
echo "Starting Ollama server..."

# E) Start the server (let it auto-detect CUDA)
unset HSA_OVERRIDE_GFX_VERSION ROCR_VISIBLE_DEVICES GGML_VK_VISIBLE_DEVICES GPU_DEVICE_ORDINAL OLLAMA_LLM_LIBRARY
export OLLAMA_HOST=127.0.0.1:11434
ollama serve |& tee ~/ollama.log     # leave running

export PATH="$HOME/opt/ollama/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/ollama/lib/ollama${LD_LIBRARY_PATH:+:
$LD_LIBRARY_PATH}"
export OLLAMA_HOST=127.0.0.1:11434
ollama pull llama3:8b
ollama run llama3:8b-p "hi"