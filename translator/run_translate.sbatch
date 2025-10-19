#!/bin/bash
#SBATCH --job-name=translate
#SBATCH --output=slurm_output/slurm-%j.out
#SBATCH --error=slurm_output/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00       
#SBATCH --partition=commons 
#SBATCH --gres=gpu:1                       

echo "Activating Python virtual environment..."
source ../.venv_tc/bin/activate

# echo "Running script 0"
# python trans_fin_gpu.py --csv data/test.csv --columns DESCRIPTION DESCRIPTION1 --out outputs/out.csv
# echo "Running script 1"
# python3 trans_fin_gpu.py --csv data/DIM_CONSOLIDATED_ACCIDENTS.csv --columns DESCRIPTION --out outputs/DIM_CONSOLIDATED_ACCIDENTS.translated.csv
echo "Running script 2"
python3 trans_fin_gpu.py --csv data/DIM_CONSOLIDATED_ACTION_PLAN.csv --columns ACTION_REQUIRED ACTIONS_TAKEN --out outputs/DIM_CONSOLIDATED_ACTION_PLAN.translated.csv
echo "Running script 3"
python3 trans_fin_gpu.py --csv data/DIM_CONSOLIDATED_CAPA_ACTION_PLAN.csv --columns ACTION_REQUIRED ACTIONS_TAKEN ACTIONS_TAKEN_SIMPLIFIED --out outputs/DIM_CONSOLIDATED_CAPA_ACTION_PLAN.translated.csv
echo "Running script 4"
python3 trans_fin_gpu.py --csv data/DIM_CONSOLIDATED_HAZARD_OBSERVATIONS.csv --columns DESCRIPTION --out outputs/DIM_CONSOLIDATED_HAZARD_OBSERVATIONS.translated.csv
echo "Running script 5"
python3 trans_fin_gpu.py --csv data/DIM_CONSOLIDATED_NEAR_MISSES.csv --columns DESCRIPTION --out outputs/DIM_CONSOLIDATED_NEAR_MISSES.translated.csv
echo "Running script 6"
python3 trans_fin_gpu.py --csv data/DIM_CONSOLIDATED_PROPERTY_DAMAGE_INCIDENT.csv --columns DAMAGED_ITEMS DAMAGE_DESCRIPTION --out outputs/DIM_CONSOLIDATED_PROPERTY_DAMAGE_INCIDENT.translated.csv

echo "All translation jobs finished."