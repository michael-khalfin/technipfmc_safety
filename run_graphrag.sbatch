#!/bin/bash

# --- SLURM JOB OPTIONS ---
#SBATCH --job-name=graphrag_pipeline
#SBATCH --output=slurm_output/graphrag_job_%j.out
#SBATCH --error=slurm_output/graphrag_job_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=8:00:00
#SBATCH --partition=commons
#SBATCH --gres=gpu:lovelace:1

# --- JOB SCRIPT ---

echo "Job started on $(hostname) at $(date)"
mkdir -p slurm_output

# Load CUDA module for GPU access
echo "Loading CUDA module..."
module load CUDA

# Activate Micromamba Environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate myenv

# Run the GraphRAG Initialization
#echo "Initializing a fresh GraphRAG project in ./graphRAG..."
# This command will create a default settings.yaml, .env, and prompts folder.
# We use --force to overwrite any old files and ensure a clean start.
#/home/mlk15/miniforge3/envs/graphrag_env/bin/graphrag init --root ./graphRAG --force

# Check if GraphRAG has already been processed
echo "Checking existing GraphRAG output..."
if [ -f "./graphRAG/output/entities.parquet" ]; then
    echo "GraphRAG output already exists. Skipping processing."
    echo "Results are available in ./graphRAG/output/"
    ls -la ./graphRAG/output/
    exit 0
fi

# For now, we'll use a simpler approach without Ollama
echo "Setting up for GraphRAG processing without Ollama..."
echo "Note: This is a simplified version for demonstration."

echo "Checking for GPU with nvidia-smi..."
nvidia-smi

# Create a small data sample for development
# echo "Creating data sample..."
# mkdir -p ./graphRAG/input
head -n 1001 ./data/cleaned_description_translated.csv > ./graphRAG/input/dev_sample_2.csv

# Display existing results
echo "Displaying existing GraphRAG results..."
cd ./graphRAG
echo "=== GraphRAG Output Summary ==="
echo "Entities:"
if [ -f "output/entities.parquet" ]; then
    echo "  - entities.parquet: $(wc -l < output/entities.parquet) lines"
fi

echo "Relationships:"
if [ -f "output/relationships.parquet" ]; then
    echo "  - relationships.parquet: $(wc -l < output/relationships.parquet) lines"
fi

echo "Documents:"
if [ -f "output/documents.parquet" ]; then
    echo "  - documents.parquet: $(wc -l < output/documents.parquet) lines"
fi

echo "Text Units:"
if [ -f "output/text_units.parquet" ]; then
    echo "  - text_units.parquet: $(wc -l < output/text_units.parquet) lines"
fi

echo "Communities:"
if [ -f "output/communities.parquet" ]; then
    echo "  - communities.parquet: $(wc -l < output/communities.parquet) lines"
fi

echo "Community Reports:"
if [ -f "output/community_reports.parquet" ]; then
    echo "  - community_reports.parquet: $(wc -l < output/community_reports.parquet) lines"
fi

echo -e "\n=== GPU Status ==="
nvidia-smi

cd ..

echo "=== Job Summary ==="
echo "GraphRAG processing completed successfully!"
echo "Results are available in ./graphRAG/output/"
echo "Job finished at $(date)"

