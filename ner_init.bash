module load Miniforge3/24.11.3-0
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate nlp
pip install -r requirements.txt
python -m spacy download en_core_web_sm