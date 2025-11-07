#!/bin/bash

# --- SLURM JOB OPTIONS ---
#SBATCH --job-name=graphrag_pipeline
#SBATCH --output=slurm_output/graphrag_job_%j.out
#SBATCH --error=slurm_output/graphrag_job_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=4:00:00
#SBATCH --partition=commons
#SBATCH --gres=gpu:lovelace:1

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

<<<<<<< HEAD
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
=======
# Activate Conda Environment
source /home/mlk15/miniforge3/etc/profile.d/conda.sh
conda activate graphrag_env
>>>>>>> 7c4c605 (here is my conda env)

# Run the GraphRAG Initialization
#echo "Initializing a fresh GraphRAG project in ./graphRAG..."
# This command will create a default settings.yaml, .env, and prompts folder.
# We use --force to overwrite any old files and ensure a clean start.
# graphrag init --root ./graphRAG --force

# Start Ollama Server in the background
echo "Starting Ollama server..."
<<<<<<< HEAD
~/bin/ollama serve &
=======
# Set OLLAMA_KEEP_ALIVE to -1 to keep the model loaded in GPU memory indefinitely.
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=4
CUDA_VISIBLE_DEVICES=0 /home/mlk15/bin/ollama serve &
>>>>>>> 7c4c605 (here is my conda env)
OLLAMA_PID=$!
echo "Ollama server started with PID: $OLLAMA_PID. Waiting for it to initialize..."
sleep 15

<<<<<<< HEAD
echo "Checking Ollama system info..."

# Create a small data sample for development
echo "Creating data sample..."
mkdir -p ./graphRAG/input
head -n 1001 ./data/cleaned_description_translated.csv > ./graphRAG/input/dev_sample.csv
=======
# Run nvidia-smi to confirm GPU is visible
echo "Pre-warming the GPU: loading the llama3:70b model..."
# We run this in the background and don't care about the output. It's just to get the model into VRAM.
/home/mlk15/bin/ollama run llama3:70b "Pre-warm" > /dev/null 2>&1 &
echo "Model loading initiated. Waiting for GPU to warm up..."
# Give it ample time to load the large 70B model.
sleep 60

echo "Checking for GPU with nvidia-smi..."
nvidia-smi

# Create a small data sample for development
# echo "Creating data sample..."
# mkdir -p ./graphRAG/input
head -n 101 ./data/cleaned_description_translated.csv > ./graphRAG/input/dev_sample_2.csv
>>>>>>> 7c4c605 (here is my conda env)

# Run the main GraphRAG pipeline
echo "Running the main GraphRAG pipeline..."
cd ./graphRAG
<<<<<<< HEAD
graphrag index
cd .. 
=======
# The command will now automatically find and use settings.yaml
/home/mlk15/miniforge3/envs/graphrag_env/bin/graphrag index
# GRAPHRAG_PID=$!

# echo -e "\n--- Begin GPU monitoring loop (every 30 seconds) ---"
# # This loop will run until the GraphRAG process finishes or the job times out
# while ps -p $GRAPHRAG_PID > /dev/null; do
#     echo -e "\n--- $(date) ---"
#     nvidia-smi
#     sleep 30
# done

cd ..
>>>>>>> 7c4c605 (here is my conda env)

# 7. Shutdown Ollama Server
echo "Shutting down Ollama server..."
kill $OLLAMA_PID
wait $OLLAMA_PID 2>/dev/null

echo "Job finished at $(date)"

