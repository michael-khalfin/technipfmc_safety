
# TechnipFMC Safety Incident Analysis Project

A comprehensive data analysis and knowledge graph generation system for safety incident data processing, featuring advanced data cleaning, visualization, and GraphRAG-based knowledge extraction capabilities.

## Project Overview

This project provides a complete pipeline for analyzing safety incident data, including:

- **Data Integration**: Merging multiple safety data sources (accidents, near misses, hazard observations)
- **Data Cleaning**: Advanced data cleaning and preprocessing with conflict resolution
- **Visualization**: Comprehensive data visualization and exploratory data analysis
- **Knowledge Graph**: GraphRAG-based knowledge graph generation for safety insights
- **Evaluation**: Multi-level graph evaluation framework for internal validation and iterative improvement
- **Translation**: Multi-language support for international safety data

## Directory Structure

```
├── src/                        # Core code and modules
├── eda/                        # Exploratory Data Analysis and Cleaning
│   ├── data_clean/             # Data cleaning and integration modules
│   ├── dataVisualizer.py       # Data visualization tools
│   ├── dataModifier.py         # Data transformation utilities
│   ├── dataFeatures.py         # Feature analysis tools
│   └── main.py                 # Main EDA execution script
├── graphRAG/                   # GraphRAG knowledge graph generation
│   ├── input/                  # Input data processing
│   ├── output/                 # Generated knowledge graph on 30,000 Incidents 
│   ├── output_1k_mistral/      # Generated knowledge graph on 1,000 Incidents (Mistral:7b-instruct)
│   ├── output_1k_phi/          # Generated knowledge graph on 1,000 Incidents (Phi3:Mini)
│   ├── prompts/                # Prompts used for graphRAG (Graph Extraction, Summarization, etc)
│   ├── extract.py              # Extracts Triplets Per Incident
│   ├── postprocess.py          # Removes (outlier) nodes from generated KG
│   └── settings.yaml           # GraphRAG configuration
├── evaluation/                 # Knowledge graph evaluation framework
│   ├── KG1/                    # Example KG folder (includes nodes.csv, edges.csv)
│   ├── KG2/                    # Second KG for comparison
│   ├── entity_consistency_eval.py   # Evaluates entity redundancy & type coherence
│   ├── link_prediction_holdout.py   # Tests predictive structure quality via link prediction
│   ├── semantic_similar_distance.py # Correlates semantic similarity with graph distance
│   └── evaluate_all.py              # Unified runner to execute all evaluation methods
├── KG_Plumber/                      # Triple extraction via ThePlumber (Docker-based)
│   ├── output/                      # Generated plumber_triples.jsonl, nodes.csv, edges.csv
│   ├── get_plumber.sh               # Script to download and set up ThePlumber locally
│   └── process_csv.py               # Sends CSV data to Plumber API for triple extraction
├── KG_spaCy/                        # Rule-based triple extraction using spaCy/Textacy
│   ├── KG_test.py                   # Extracts subject–verb–object triples from text
│   └── triple_clean.py              # Cleans and normalizes extracted triples
├── incident-embedding-analysis/     # Incident-level embedding & comparison
│   ├── text_embedding.py            # Text-based incident embedding
│   ├── graph_embedding.py           # Graph-based (Node2Vec) embedding
│   ├── transe_embedding.py          # Knowledge graph embedding (TransE)
│   ├── requirements.txt             # Dependencies for embedding & analysis
│   └── analysis/                    # Embedding alignment & analysis
│       ├── align.py                 # Align embeddings by incident ID
│       ├── correlation.py           # Correlation analysis
│       └── heatmap.py               # Similarity heatmap visualization
├── translator/                      # Multi-language translation tools
│   ├── csv_translator_m2m100_gpu.py    # Translates CSV columns using M2M100 with GPU
│   └── run_translate.sbatch            # SLURM script for translation on HPC
├── visual_dashboard/             # Interactive Streamlit dashboard
│   └── dashboard/                # Dashboard application
│       ├── app.py                # Main Streamlit application
│       ├── handler/              # Chart generation handlers
│       │   ├── handler.py        # Main chart handler with filtering
│       │   └── graph_makers.py   # Chart generation functions
│       └── preprocessing/        # Data preprocessing utilities
├── data/                       # Safety incident data (gitignored)
└── viz.py                      # Graph visualization utilities
```

## Key Features

### 1. Data Integration & Cleaning
- **Multi-source Integration**: Combines accidents, near misses, hazard observations, and action plans
- **Conflict Resolution**: Intelligent handling of data conflicts during merging
- **Data Quality Analysis**: Comprehensive data quality assessment and reporting
- **Column Analysis**: Advanced column compatibility analysis for safe merging

### 2. Data Visualization
- **Interactive Dashboards**: Comprehensive data type and cardinality analysis
- **Missing Value Analysis**: Detailed missing data visualization
- **Correlation Analysis**: Heatmaps and correlation pair identification
- **Text Analysis**: N-gram analysis and word clouds for incident descriptions
- **Variance Analysis**: Feature variance visualization for model selection

### 2.a Interactive Web Dashboard
- **Streamlit-based Dashboard**: Real-time interactive web interface for safety event analysis
- **Advanced Filtering**: Multi-dimensional filtering by year/month, incident type, status, risk color, GBU, location, severity, and likelihood
- **Comprehensive Visualizations**: 
  - Statistical summary with key metrics (total incidents, high risk percentage, average severity/likelihood, open actions)
  - Temporal distribution charts (by year or month)
  - Top locations and units analysis
  - Incident type and impact type distributions (pie charts)
  - Risk color distribution by severity (stacked bar charts)
  - Top causes/hazards analysis
  - Severity vs Likelihood heatmaps
  - Word cloud visualization of incident titles
  - Event cluster visualization with t-SNE
- **Dynamic Data Filtering**: All visualizations update in real-time based on selected filters
- **Data Quality Tracking**: Built-in data quality metrics for each visualization

### 3. Knowledge Graph Generation
- **GraphRAG Integration**: Automated knowledge graph creation from safety data
- **Entity Extraction**: Automatic extraction of entities and relationships
- **Interactive Visualization**: NetworkX and PyVis-based graph visualization
- **Community Detection**: Automatic community identification in safety networks

### 4. Evaluation Framework 
#### **Method 1 – Entity Consistency Evaluation**
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`  
- **Clustering Method**: *K-Means*  
- **Metrics**:  
  - **Unique Entity Rate** — measures redundancy within entity clusters  
  - **Type Consistency Rate** — checks semantic alignment across entity types  

#### **Method 2 – Link Prediction Holdout**  
- **Procedure**:
  - Randomly occlude *p%* of real edges as the **positive test set**  
  - Sample an equal number of **negative samples** from non-existing edges  
  - Score candidate edges using a set of structural or embedding-based heuristics:
    - *Adamic-Adar*, *Jaccard*, *Resource Allocation*, *Personalized PageRank*, *Embedded Dot Product*  
  - Evaluate the model’s ability to distinguish real relationships from spurious ones using precision–recall and ranking metrics  
- **Metrics**: `PR-AUC`, `ROC-AUC`, `P@K`, `R@K`, `MAP`
  
#### **Method 3 – Semantic Similarity vs Graph Distance**
- **Metrics**: `Spearman’s ρ (rho)` and `p-value`  
- **Goal**: Verify whether “the more semantically similar the nodes are, the closer they are in the graph.”  
- **Steps**:
  - Randomly sample node pairs  
  - Compute semantic similarity `sim(label_u, label_v)` using embedding cosine or token overlap  
  - Compute shortest path distance `dist_G(u, v)` on the knowledge graph  
  - Calculate the **Spearman correlation** between the two; a significant **negative correlation** (i.e., larger |ρ|) implies better *semantic alignment* between textual meaning and graph topology  

### 5. Multi-language Support
- **Translation Pipeline**: M2M100-based translation for international data
- **Language Detection**: Automatic source language detection
- **GPU Acceleration**: CUDA-optimized translation processing

### 6. Incident Embedding & Representation Analysis
This module constructs **incident-level embeddings** from multiple perspectives and analyzes their relationships.
* **Text Embedding**: Semantic representations from incident descriptions
* **Graph Embedding**: Structural representations using Node2Vec
* **TransE Embedding**: Relational representations from incident knowledge graphs

The embeddings are aligned and compared using:
* Pairwise similarity correlation (Pearson / Spearman)
* Clustered cosine-similarity heatmaps for qualitative inspection

## Setup Instructions

### 1. Environment Setup
```bash
# Clone and navigate to project directory
cd technipfmc_safety

# Copy data from shared location
cp -r /projects/dsci435/fmcsafetyevents_fa25/data .

# Create virtual environment
python3 -m venv .venv 
source .venv/bin/activate

# Install dependencies
pip install -U pip
pip install -r requirements.txt
```

### 1.a GraphRag Setup
```bash
# Load Modules
module load foss/2023b        
module load Python/3.11.5     # loads Python 3.11 (good for GraphRAG)
module load CUDA/12.4.0
module load Miniforge3/24.11.3-0

# Create Conda Environment
conda create -n graphrag_env python=3.11
source $(conda info --base)/etc/profile.d/conda.sh
conda activate graphrag_env
pip install -r requirements.txt
```
### 1.b Downloading Ollama
```bash

# Create and Opt Directory (Uncomment below if disk quota exceeded)
export OLLAMA_DIR="$HOME/opt/ollama"
# export OLLAMA_DIR="/scratch/$USER/opt/ollama"  
mkdir -p "$OLLAMA_DIR" 

# Download Ollama Tarbell
cd $OLLAMA_DIR
curl -LO https://ollama.com/download/ollama-linux-amd64.tgz

# Unpack Tarbell
tar -xvzf ollama-linux-amd64.tgz

# Activate Ollama 
chmod +x $OLLAMA_DIR/bin/ollama

# (TENTATIVE TO CHANGE) Run and PrePull Models
./bin/ollama serve

# Models
"$OLLAMA_DIR/bin/ollama" pull llama3:8b
"$OLLAMA_DIR/bin/ollama" pull nomic-embed-text:latest
```

### 1.c Plumber Setup (LOCAL ONLY SETUP)
```bash
# Set get_plumber.sh executable
chmod +x KG_Plumber\get_plumber.sh

# Run Executable
./get_plumber.sh

# Set Up Docker
cd KG_Plumber\plumber
docker-compose up -d
```

### 2. Resource Allocation (NOTS Cluster)
```bash
# Submit job for resource allocation
sbatch submit.sbatch
```

### 3. Python Version Requirements
```bash
# Load required modules (GraphRAG requires Python 3.10-3.12)
module load GCCcore/13.2.0
module load Python/3.11.5
module load CUDA/12.4.0

# Verify Python version
python3 --version
```

## Usage

### Data Cleaning and EDA
```bash
# Run comprehensive data analysis
python3 eda/main.py

# Run with description-only processing
python3 eda/main.py --descriptions
```

### Knowledge Graph Generation
```bash
# Update requirements
pip freeze > requirements.txt

# Run GraphRAG indexing
run_graphrag.sbatch
```

### GraphRAG Postprocessing & Incident Triplet Extraction
After GraphRAG finishes, it will write parquet outputs under `graphRAG/output/` ( `entities.parquet`, `relationships.parquet`, `text_units.parquet`, `documents.parquet`).  
Use the graphRAG/postprocess.py to clean the graph by dropping metadata-heavy nodes/edges and graphRAG/extractor.py to extract triples per incident:

```bash
python graphRAG/postprocess.py -A -D
```
- Drops META nodes whose titles start with `META[`.
- Removes very high-degree nodes (top ~10% by degree, clears "noisy" nodes).
- If `-A/--allowed` is set, keeps only entities whose type is in `ALLOWED_TYPES`.
- If `-D/--drop-labels` is set, removes relationships whose label is in `DROP_RELATIONS`.
- Writes filtered outputs to:
  - `graphRAG/output/entities_filtered.parquet`
  - `graphRAG/output/relationships_filtered.parquet`

```bash
python graphRAG/extract.py
```
- Generates a flattened CSV of triples per incident at:
  - `graphRAG/output/incident_triples.csv`
- Each row includes: `incident_id`, `source`, `description` (relation), `target`, `text_unit_id`, `relationship_id`, and `document_id`.
- Prints the average number of triplets per incident. Can enable detailed printing per incident by editing `DESCRIPTIVE` in `graphRAG/extract.py`.

### Graph Visualization
```bash
# Generate interactive graph visualizations
python viz.py
```

### Data Translation
```bash
# Translate CSV columns to English
python translator/csv_translator_m2m100_gpu.py --csv input.csv --columns "TITLE,DESCRIPTION" --out output.csv
```

### Interactive Web Dashboard
```bash
# Launch the Streamlit dashboard
streamlit run visual_dashboard/dashboard/app.py
```

The dashboard provides an interactive web interface for exploring safety incident data with:
- **Sidebar Filters**: Filter data by year/month range, incident types, status, impact type, risk color, GBU, workplace country, severity, and likelihood
- **Real-time Updates**: All charts update automatically when filters are changed
- **Multiple Visualizations**: 
  - Statistical summary metrics
  - Temporal distribution (yearly or monthly)
  - Top locations and units
  - Incident type and impact type distributions
  - Risk color analysis
  - Top causes/hazards
  - Severity vs Likelihood heatmap
  - Word cloud of incident titles
  - Event cluster visualization with customizable sampling ratio

**Note**: The dashboard requires `merged_incidents_tsne.csv` in the `data/` directory. Ensure this file exists before running the dashboard.

## Module Documentation

### Data Cleaning Modules (`eda/data_clean/`)

- **`DataLoader`**: Main data loading and integration class
- **`ColumnAnalyzer`**: Column compatibility analysis for safe merging
- **`Coalescer`**: Data merging with conflict resolution
- **`utils.py`**: Utility functions for data analysis
- **`report.py`**: Data quality reporting and analysis

### Visualization Modules (`eda/`)

- **`DataVisualizer`**: Comprehensive data visualization class
- **`DataFormatter`**: Data formatting and analysis utilities
- **`dataModifier`**: Data transformation and cleaning
- **`dataFeatures`**: Feature analysis and dataset summaries

### Translation Module (`translator/`)

- **`csv_translator_m2m100_gpu.py`**: Multi-language CSV translation with GPU acceleration

### Graph Visualization (`viz.py`)

- Interactive graph visualization using NetworkX and PyVis
- Static and dynamic graph generation
- Community detection and analysis

### Interactive Web Dashboard (`visual_dashboard/`)

- **`app.py`**: Main Streamlit application that provides the web interface
  - Sidebar filters for data filtering
  - Multiple chart visualizations
  - Real-time data updates
  
- **`handler/handler.py`**: Chart generation handler
  - Data loading and caching
  - Filter application logic
  - Chart type routing and generation
  
- **`handler/graph_makers.py`**: Chart generation functions
  - Statistical summary generation
  - Bar charts (all categories, top K, stacked)
  - Pie charts
  - Heatmaps
  - Word clouds
  - Temporal distribution charts
  - Event cluster visualization (t-SNE based)
  - Data quality tracking for each visualization

**Key Features**:
- Supports filtering by numerical ranges (min/max) and categorical values (single or multiple)
- All charts are generated using Plotly for interactive exploration
- Data quality metrics are tracked and can be returned with visualizations
- Efficient data caching to improve performance

### Plumber-Based Extraction (`KG_Plumber/`)

- **`get_plumber.sh`**:  Script that clones the upstream [ThePlumber](https://github.com/YaserJaradeh/ThePlumber) repository into `KG_Plumber/plumber` using optional `PLUMBER_REPO_URL` and `PLUMBER_BRANCH` overrides.
- **`process_csv.py`**: Sends incident descriptions to a running Plumber API (`http://127.0.0.1:5000` by default), retries failures, and incrementally writes outputs to `plumber_triples.jsonl`, `nodes.csv`, and `edges.csv` within the `KG_Plumber/outputs` directory.
- **Usage**: Start the Plumber service (Docker), update `CSV_PATH` if needed, then run `python KG_Plumber/process_csv.py` to materialize triples and derived graph artifacts.

### SpaCy Triple Extraction (`KG_spaCy/`)

- **`KG_test.py`**: Prototyping script that loads `cleaned_description_translated.csv`, extracts subject-verb-object triples with `textacy`/spaCy, and exports raw edges to `knowledge_graph_edges.csv`.
- **`triple_clean.py`**: Cleans the raw triples by normalizing entities/relations, filtering  long phrases, and producing`nodes.csv` and `edges.csv`.
- **Usage**: Run `python KG_spaCy/KG_test.py` to generate initial triples, then `python KG_spaCy/triple_clean.py` to normalize them.

### Evaluation Framework (`evaluation/`)
- **`entity_consistency_eval.py`**:  Evaluates semantic redundancy and type coherence among extracted entities.  
- **`link_prediction_holdout.py`**:  Assesses graph structural quality through link prediction.
- **`semantic_similar_distance.py`**:  Tests whether semantically similar nodes are topologically close.  
- **`evaluate_all.py`**:Unified runner that executes all evaluation modules sequentially on a given knowledge graph directory.  

how to run:
```bash
python evaluation/evaluate_all.py KG1
```

## Configuration

### GraphRAG Settings
Edit `graphRAG/settings.yaml` to configure:
- Entity extraction parameters
- Relationship detection settings
- Community detection thresholds
- Output formatting options

### Visualization Settings
Modify visualization parameters in `viz.py`:
- `MIN_EDGE_WEIGHT`: Filter weak relationships
- `TOP_N_NODES`: Limit graph size for performance
- `SEED`: Random seed for consistent layouts

## Output Files

### Data Analysis Outputs
- `data/cleaned_data.csv`: Cleaned and integrated dataset
- `data/cleaned_description_translated.csv`: Translated descriptions
- `data/merged_incidents_tsne.csv`: Merged incidents data with t-SNE coordinates (required for dashboard)
- `eda/visualization/`: Generated visualization files

### Knowledge Graph Outputs
- `graphRAG/output/` Holds KG generated from 30,000 Incidents
- `graphRAG/output_1k_mistral/` Holds KG generated from 1,000 Incidents (mistral:7b-intruct)
- `graphRAG/output_1k_phi/` Holds KG generated from 1,000 Incidents (phi3:mini)
- `graphRAG/output/entities.parquet`: Extracted entities
- `graphRAG/output/relationships.parquet`: Relationship data
- `graphRAG/output/_viz/`: Interactive graph visualizations

## Dependencies

Key Python packages:
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `matplotlib`, `seaborn`: Data visualization
- `networkx`, `pyvis`: Graph analysis and visualization
- `transformers`: M2M100 translation model
- `graphrag`: Knowledge graph generation
- `scikit-learn`: Machine learning utilities
- `streamlit`: Interactive web dashboard framework
- `plotly`: Interactive chart generation for dashboard
- `wordcloud`: Word cloud generation for text analysis

## Contributing

1. Follow the existing code structure and documentation standards
2. Add comprehensive docstrings to all new functions and classes
3. Update this README when adding new features
4. Test all changes with the provided data samples

## License

This project is part of the TechnipFMC safety data analysis initiative.
