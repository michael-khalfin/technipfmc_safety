#!/bin/bash
#SBATCH --job-name=graphrag_pipeline
#SBATCH --output=slurm_output/graphrag_job_%j.out
#SBATCH --error=slurm_output/graphrag_job_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=10G
#SBATCH --time=04:20:00
#SBATCH --partition=commons
#SBATCH --gres=gpu:lovelace:1

echo "Job started on $(hostname) at $(date)"
mkdir -p slurm_output

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} (Slurm remaps your GPU to 0 inside the job)"

# ---------------------
# CONFIGURATION
# ---------------------
export DATA_DIR="/scratch/$USER"
export LOG_PATH="/home/$USER/COMP449/technipfmc_safety"
mkdir -p "$DATA_DIR/opt/ollama" "$DATA_DIR/logs"
export OLLAMA_MODELS="$DATA_DIR/.ollama"
mkdir -p "$OLLAMA_MODELS"

# Trick Ollama only - override its config path, not global HOME
export OLLAMA_HOME="$OLLAMA_MODELS"
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_NUM_PARALLEL=16
export OLLAMA_MAX_QUEUE=32
export OLLAMA_KEEP_ALIVE=0


# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate graphrag_env
export PATH="$CONDA_PREFIX/bin:$DATA_DIR/opt/ollama/bin:$PATH"
export LD_LIBRARY_PATH="$DATA_DIR/opt/ollama/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

unset HSA_OVERRIDE_GFX_VERSION ROCR_VISIBLE_DEVICES GGML_VK_VISIBLE_DEVICES GPU_DEVICE_ORDINAL OLLAMA_LLM_LIBRARY

echo "Checking for GPU with nvidia-smi..."
nvidia-smi -L || true
nvidia-smi || true

# ---------------------
# START OLLAMA SERVER
# ---------------------

# Flush old log
: > "$LOG_PATH/ollama.log"

echo "Starting Ollama server (legacy v0.12.x mode)..."
nohup "$DATA_DIR/opt/ollama/bin/ollama" serve > "$LOG_PATH/ollama.log" 2>&1 &
OLLAMA_PID=$!

# Wait for server to come online
for i in {1..10}; do
    if curl -s http://127.0.0.1:11434/api/version > /dev/null 2>&1; then
        echo "Ollama server is up!"
        break
    fi
    echo "Waiting for Ollama to start ($i/30)..."
    sleep 2
done
if ! curl -s http://127.0.0.1:11434/api/version > /dev/null 2>&1; then
    echo "Ollama failed to start. Recent log:"
    tail -n 40 "$DATA_DIR/ollama.log"
    exit 1
fi

# ---------------------
# PULL MODELS
# ---------------------
CHAT_MODEL="mistral"
EMBED_MODEL="nomic-embed-text"

if ! "$DATA_DIR/opt/ollama/bin/ollama" list | grep -q "$CHAT_MODEL"; then
  echo "Pulling chat model $CHAT_MODEL..."
  "$DATA_DIR/opt/ollama/bin/ollama" pull "$CHAT_MODEL" || { echo "Failed to pull $CHAT_MODEL"; kill ${OLLAMA_PID} 2>/dev/null; exit 1; }
fi
if ! "$DATA_DIR/opt/ollama/bin/ollama" list | grep -q "$EMBED_MODEL"; then
  echo "Pulling embedding model $EMBED_MODEL..."
  "$DATA_DIR/opt/ollama/bin/ollama" pull "$EMBED_MODEL" || { echo "Failed to pull $EMBED_MODEL"; kill ${OLLAMA_PID} 2>/dev/null; exit 1; }
fi
# ---------------------
# GRAPHRAG PIPELINE
# ---------------------
echo "Preparing Input File"
mkdir -p ./graphRAG/input
# head -n 1001 ./data/cleaned_description_translated.csv > ./graphRAG/input/dev_sample.csv

echo "Preparing GraphRAG workspace..."
mkdir -p ./graphRAG/input
if [ ! -f ./graphRAG/settings.yaml ]; then
  python -m graphrag init --root ./graphRAG
fi

echo "Running GraphRAG index..."
pushd ./graphRAG >/dev/null
python -m graphrag prompt-tune --root /home/cdm8/COMP449/technipfmc_safety/graphRAG --config /home/cdm8/COMP449/technipfmc_safety/graphRAG/settings.yaml
popd >/dev/null

# ---------------------
# CLEANUP
# ---------------------
echo "Shutting down Ollama server..."
kill ${OLLAMA_PID} 2>/dev/null || true
wait ${OLLAMA_PID} 2>/dev/null || true

echo "Job finished at $(date)"