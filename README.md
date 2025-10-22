
# TechnipFMC Safety Incident Analysis Project

A comprehensive data analysis and knowledge graph generation system for safety incident data processing, featuring advanced data cleaning, visualization, and GraphRAG-based knowledge extraction capabilities.

## Project Overview

This project provides a complete pipeline for analyzing safety incident data, including:

- **Data Integration**: Merging multiple safety data sources (accidents, near misses, hazard observations)
- **Data Cleaning**: Advanced data cleaning and preprocessing with conflict resolution
- **Visualization**: Comprehensive data visualization and exploratory data analysis
- **Knowledge Graph**: GraphRAG-based knowledge graph generation for safety insights
- **Translation**: Multi-language support for international safety data


python knowlege_graph\process_csv.py "C:\Users\carda\projects\449COMP\technipfmc_safety\data\cleaned_description_translated.csv" --text-column text --max-rows 1000 --output knowlege_graph\output\plumber_triples.jsonl --nodes-csv knowlege_graph\output\nodes.csv --edges-csv knowlege_graph\output\edges.csv --extractor OpenIE --resolver dummy --warmup 5 --max-workers 2 --retries 3 --retry-wait 8 --timeout 180 --log-every 20
[warmup] sending 5 sequential requests before parallel run...


## Directory Structure

```
├── src/                    # Core code and modules
├── eda/                    # Exploratory Data Analysis and Data Cleaning
│   ├── data_clean/         # Data cleaning and integration modules
│   ├── dataVisualizer.py   # Data visualization tools
│   ├── dataModifier.py     # Data transformation utilities
│   ├── dataFeatures.py     # Feature analysis tools
│   └── main.py            # Main EDA execution script
├── graphRAG/              # GraphRAG knowledge graph generation
│   ├── input/             # Input data processing
│   ├── output/            # Generated knowledge graphs
│   └── settings.yaml      # GraphRAG configuration
├── translator/            # Multi-language translation tools
├── data/                  # Safety incident data (gitignored)
└── viz.py                 # Graph visualization utilities
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

### 3. Knowledge Graph Generation
- **GraphRAG Integration**: Automated knowledge graph creation from safety data
- **Entity Extraction**: Automatic extraction of entities and relationships
- **Interactive Visualization**: NetworkX and PyVis-based graph visualization
- **Community Detection**: Automatic community identification in safety networks

### 4. Multi-language Support
- **Translation Pipeline**: M2M100-based translation for international data
- **Language Detection**: Automatic source language detection
- **GPU Acceleration**: CUDA-optimized translation processing

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
- `eda/visualization/`: Generated visualization files

### Knowledge Graph Outputs
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

## Contributing

1. Follow the existing code structure and documentation standards
2. Add comprehensive docstrings to all new functions and classes
3. Update this README when adding new features
4. Test all changes with the provided data samples

## License

This project is part of the TechnipFMC safety data analysis initiative.
