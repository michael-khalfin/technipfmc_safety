#!/bin/bash

# --- SLURM JOB OPTIONS ---
#SBATCH --job-name=graphrag_pipeline
#SBATCH --output=slurm_output/graphrag_job_%j.out
#SBATCH --error=slurm_output/graphrag_job_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=08:00:00
#SBATCH --partition=commons
#SBATCH --gres=gpu:1

# --- JOB SCRIPT ---

echo "Job started on $(hostname) at $(date)"
# Ensure output directory exists if using --output/--error paths with a folder
mkdir -p slurm_output

# Load CUDA module for GPU access
echo "Loading CUDA module..."
module load CUDA

# Run nvidia-smi to confirm GPU is visible
echo "Checking for GPU with nvidia-smi..."
nvidia-smi

# Use conda environment (create if missing), no venv fallback
echo "Loading Miniforge3 module..."
module load Miniforge3/24.11.3-0
echo "Initializing conda..."
eval "$("${CONDA_EXE:-conda}" shell.bash hook)" || {
  CONDA_BASE="$(conda info --base 2>/dev/null)" && \
  . "$CONDA_BASE/etc/profile.d/conda.sh"
}

ENV_NAME="${ENV_NAME:-graphrag_env}"
PY_VER="${PY_VER:-3.11}"
export CONDA_ENVS_PATH="/scratch/$USER/conda/envs"
export CONDA_PKGS_DIRS="/scratch/$USER/conda/pkgs"
mkdir -p "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS"

ENV_PREFIX="$CONDA_ENVS_PATH/$ENV_NAME"
echo "Ensuring conda env at '$ENV_PREFIX' exists (Python $PY_VER)..."
if [ ! -d "$ENV_PREFIX" ]; then
  conda create -y -p "$ENV_PREFIX" "python=$PY_VER"
  CREATE_STATUS=$?
  if [ $CREATE_STATUS -ne 0 ]; then
    echo "ERROR: Failed to create conda env at '$ENV_PREFIX'" >&2
    exit $CREATE_STATUS
  fi
  conda activate "$ENV_PREFIX"
  python -m pip install --upgrade pip
  pip install -r "$SLURM_SUBMIT_DIR/requirements.txt"
else
  conda activate "$ENV_PREFIX"
fi
python -V
which python
which graphrag || true

# Run the GraphRAG Initialization
#echo "Initializing a fresh GraphRAG project in ./graphRAG..."
# This command will create a default settings.yaml, .env, and prompts folder.
# We use --force to overwrite any old files and ensure a clean start.
#/home/mlk15/miniforge3/envs/graphrag_env/bin/graphrag init --root ./graphRAG --force

# Start Ollama Server (fixed paths under /scratch)
echo "Starting Ollama server..."
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MODELS="/scratch/$USER/.ollama"

OLLAMA_BIN="/scratch/$USER/bin/ollama"
if [ ! -x "$OLLAMA_BIN" ]; then
  echo "ERROR: Ollama binary not found at $OLLAMA_BIN" >&2
  echo "Ensure you extracted ollama to /scratch/$USER/bin/ollama and made it executable." >&2
  exit 1
fi

"$OLLAMA_BIN" --version || true
"$OLLAMA_BIN" serve &
OLLAMA_PID=$!
sleep 20

# Create a 1k-row sample and convert to JSON-format expected by GraphRAG
echo "Creating 1k-row sample and converting to JSON..."
mkdir -p ./graphRAG/input
head -n 1001 ./data/cleaned_description_translated.csv > ./graphRAG/input/dev_sample.csv
pushd ./graphRAG/input >/dev/null
python convert_csv.py
popd >/dev/null

# Run the main GraphRAG pipeline
echo "Running the main GraphRAG pipeline..."
cd ./graphRAG
# The command will now automatically find and use settings.yaml
srun --ntasks=1 graphrag index
cd ..

# 7. Shutdown Ollama Server
echo "Shutting down Ollama server..."
kill $OLLAMA_PID
wait $OLLAMA_PID 2>/dev/null

echo "Job finished at $(date)"

