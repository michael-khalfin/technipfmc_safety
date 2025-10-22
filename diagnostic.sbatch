#!/bin/bash

# --- SLURM JOB OPTIONS for GPU Monitoring ---
#SBATCH --job-name=gpu_monitor
#SBATCH --output=slurm_output/gpu_monitor_%j.out
#SBATCH --error=slurm_output/gpu_monitor_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=01:00:00 # One hour is enough for this test
#SBATCH --partition=commons
#SBATCH --gres=gpu:lovelace:1

# --- DIAGNOSTIC SCRIPT ---

echo "--- GPU Monitoring Job Started: $(date) ---"

# 1. Setup Environment
echo -e "\n--- Loading modules and activating environment ---"
module load CUDA
source /home/mlk15/miniforge3/etc/profile.d/conda.sh
conda activate graphrag_env

# 2. Start Ollama Server
echo -e "\n--- Starting Ollama server ---"
# We are NOT setting OLLAMA_KEEP_ALIVE yet, to observe the default behavior
CUDA_VISIBLE_DEVICES=0 /home/mlk15/bin/ollama serve &
OLLAMA_PID=$!
sleep 15

# # 3. Run GraphRAG in the background
echo -e "\n--- Starting GraphRAG pipeline in the background ---"
/home/mlk15/miniforge3/envs/graphrag_env/bin/graphrag ./graphRAG index &
GRAPHRAG_PID=$!

# 4. Monitor GPU Usage
echo -e "\n--- Begin GPU monitoring loop (every 30 seconds) ---"
# This loop will run until the GraphRAG process finishes or the job times out
while ps -p $GRAPHRAG_PID > /dev/null; do
    echo -e "\n--- $(date) ---"
    nvidia-smi
    sleep 30
done

echo -e "\n--- GraphRAG process finished. Final GPU state: ---"
nvidia-smi

# 5. Cleanup
echo -e "\n--- Shutting down Ollama server ---"
kill $OLLAMA_PID
wait $OLLAMA_PID 2>/dev/null

echo "--- Monitoring Job Finished: $(date) ---"