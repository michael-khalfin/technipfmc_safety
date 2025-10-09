
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


# Setup for Conda & Connection To Ollama
1. module load Miniforge3/24.11.3-0
2. conda create -n nlp python=3.11 -y
3. conda activate nlp
4. pip install -r requirements.txt
5. (FOR NER): python -m spacy download en_core_web_sm
6. Download Ollam External to NOTS SYSTEM
7. (For Windows Only):
    8. In local powershell, ollama serve
    9. ssh -v -R 11434:127.0.0.1:11434 <netid>@nots.rice.edu
    10. Verify in NOTS, curl http://localhost:11434/api/tags





## Resource Allocation on NOTS
To Submit Request For Resources: 
    sbatch submit.sbatch

# Data Cleaning
In order to view the EDA, you can run the following 'python3 eda/main.py'


