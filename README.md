
## Directory Structure
- `src/` – core code and modules
- `eda/` – EDA and general Data Cleaning
- `data/` – Incident Data (uncommited by gitignore)

## Setup
1. cp -r /projects/dsci435/fmcsafetyevents_fa25/data . 
2. python3 -m venv .venv 
3. source .venv/bin/activate
4. pip install -U pip
4. pip install -r requirements.txt

## Resource Allocation on NOTS
To Submit Request For Resources: 
    sbatch submit.sbatch

# Data Cleaning
In order to view the EDA, you can run the following 'python3 eda/main.py'


# Change to python 3.11.5 (because graphRAG can only run between version 3.10-3.12)
module load GCCcore/13.2.0
module load Python/3.11.5
# check version
python3 --version  

# update requirements
pip freeze > requirements.txt
