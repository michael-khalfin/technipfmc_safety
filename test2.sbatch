#!/bin/bash
#SBATCH --job-name=graphrag_pipeline
#SBATCH --output=slurm_output/test2_job_%j.out
#SBATCH --error=slurm_output/test2_job_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=04:00:00
#SBATCH --partition=commons
#SBATCH --gres=gpu:lovelace:1

# --- JOB SCRIPT ---

echo "Job started on $(hostname) at $(date)"
mkdir -p slurm_output

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}  (Slurm remaps your GPU to 0 inside the job)"

# --- Modules / Conda (adjust if your env differs) ---
module purge
module load Miniforge3/24.11.3-0 || true

# Initialize conda and activate your env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate graphrag_env
export PATH="$CONDA_PREFIX/bin:$PATH"

# --- Ollama: user-space install & GPU env ---
# Use your user-space Ollama (installed earlier under $HOME/opt/ollama)
export PATH="$HOME/opt/ollama/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/ollama/lib/ollama${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Clean any AMD/ROCm/Vulkan leftovers; let Ollama auto-detect CUDA
unset HSA_OVERRIDE_GFX_VERSION ROCR_VISIBLE_DEVICES GGML_VK_VISIBLE_DEVICES GPU_DEVICE_ORDINAL OLLAMA_LLM_LIBRARY
export OLLAMA_HOST=127.0.0.1:11434

# --- Sanity check GPU ---
echo "Checking for GPU with nvidia-smi..."
nvidia-smi -L || true
nvidia-smi || true

# --- Start Ollama server in background on localhost ---
echo "Starting Ollama server..."
# Keep it simple on a single L40S
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
nohup "$HOME/opt/ollama/bin/ollama" serve > "$HOME/ollama.log" 2>&1 &
OLLAMA_PID=$!

# Wait for readiness (max ~60s)
echo -n "Waiting for Ollama to become ready"
for i in {1..60}; do
  if curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo " ... ready."
    break
  fi
  echo -n "."
  sleep 1
done

# If not ready, bail with useful log tail
if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo -e "\nOllama did not become ready. Recent log:"
  tail -n 100 "$HOME/ollama.log" || true
  kill ${OLLAMA_PID} 2>/dev/null || true
  exit 1
fi

echo "Preparing Input File"
mkdir -p ./graphRAG/input
head -n 1001 ./data/cleaned_description_translated.csv > ./graphRAG/input/dev_sample.csv

# Start the server and stream logs to both file and SLURM stdout
# (Note: $! will be the PID of 'tee'; we’ll find the server PID separately)
stdbuf -oL -eL "$HOME/opt/ollama/bin/ollama" serve 2>&1 | tee "$HOME/ollama.log" &
TEE_PID=$!
sleep 1
OLLAMA_PID=$(pgrep -u "$USER" -f 'ollama serve' | head -n1)


# --- GraphRAG steps ---
echo "Preparing GraphRAG workspace..."
mkdir -p ./graphRAG/input
# Initialize (idempotent)
graphrag init --root ./graphRAG --force

# Sample data for dev (adjust path/file as needed)
if [ -f ./data/cleaned_data.csv ]; then
  head -n 1001 ./data/cleaned_data.csv > ./graphRAG/input/dev_sample.csv
fi

# --- Step 3: Pre-pull the exact models you'll use ---
# (Do this AFTER the server is ready, BEFORE running GraphRAG)
echo "Pre-pulling Ollama models..."
"$HOME/opt/ollama/bin/ollama" pull llama3:8b
"$HOME/opt/ollama/bin/ollama" pull nomic-embed-text:latest

# Optional: a tiny warm-up so the backend initializes once
export OLLAMA_KEEP_ALIVE=5m
"$HOME/opt/ollama/bin/ollama" run llama3:8b -p "hi" >/dev/null 2>&1 || true
sleep 2

# --- Step 4: Verify CUDA backend (fail fast if CPU-only) ---
echo "Verifying CUDA backend in ollama.log..."
if grep -qi 'library=cpu' "$HOME/ollama.log"; then
  echo "ERROR: Ollama fell back to CPU. Recent log:"
  grep -iE 'inference compute|cuda|vram|library=' "$HOME/ollama.log" | tail -n 80 || true
  exit 2
fi

# Positive signal (optional)
grep -qiE 'cuda_v(12|13)' "$HOME/ollama.log" && echo "CUDA backend detected." || echo "WARNING: Could not confirm CUDA string yet."

echo "Running the main GraphRAG pipeline..."
pushd ./graphRAG >/dev/null
graphrag index
popd >/dev/null

# --- Cleanup ---
echo "Shutting down Ollama server..."
kill ${OLLAMA_PID} 2>/dev/null || true
wait ${OLLAMA_PID} 2>/dev/null || true

echo "Job finished at $(date)"