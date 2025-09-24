
## Directory Structure
- `src/` – core code and modules
- `notebooks/` – Jupyter notebooks for EDA and experiments
- `data/` – Incident Data (uncommited by gitignore)

## Resource Allocation on NOTS
To Submit Request For Resources: 
    sbatch submit.sbatch


## Setup
1. cp -r /projects/dsci435/fmcsafetyevents_fa25/data . 
2. python3 -m venv .venv 
3. source .venv/bin/activate
4. pip install -U pip
4. pip install -r requirements.txt