# Incident Embedding Construction and Analysis

This project builds and analyzes **incident-level embeddings** using three complementary approaches:

* **Text embedding** (semantic information)
* **Graph embedding (Node2Vec)** (structural information)
* **Knowledge graph embedding (TransE)** (relational information)

The embeddings are aligned and compared through **correlation analysis** and **similarity heatmap visualization**.

Due to data privacy constraints, raw data and embeddings are not included.

---

## Project Structure

```text
incident-embedding-analysis/
├── text_embedding.py          # Text-based incident embedding
├── graph_embedding.py         # Graph-based (Node2Vec) incident embedding
├── transe_embedding.py        # TransE-based incident embedding
│
├── analysis/
│   ├── align.py               # Align embeddings by incident ID
│   ├── correlation.py         # Correlation analysis between embeddings
│   └── heatmap.py             # Similarity heatmap visualization
│
├── requirements.txt
└── README.md
```

---

## Required Input Files

The following input files are required locally:

### 1. Incident data

```text
incident_triples.csv
```

Used by:

* `graph_embedding.py`
* `transe_embedding.py`

### 2. Incident text descriptions

```text
cleaned_description_translated.csv
```

Required columns:

* `text`
* `RECORD_NO_LOSS_POTENTIAL` (incident ID)

### 3. Metadata (for visualization only)

```text
cleaned_data_without_unnecessary_features.csv
```

Required columns:

* `RECORD_NO_LOSS_POTENTIAL`
* `SEVERITY_VALUE`

---

## Embedding Generation

### Text Embedding

Encodes incident descriptions using a sentence transformer and reduces dimensionality with PCA.

```bash
python text_embedding.py
```

Output:

```text
event_text_embedding.pkl
```

---

### Graph Embedding (Node2Vec)

Constructs an incident graph and learns structural embeddings using Node2Vec.

```bash
python graph_embedding.py
```

Output:

```text
event_graph_embedding.pkl
```

---

### TransE Embedding

Cleans and normalizes relations, then trains a TransE model on the incident knowledge graph.

```bash
python transe_embedding.py
```

Output:

```text
event_transe_embedding.pkl
```

---

## Analysis

### Step 1: Align embeddings

Align all embeddings to a common set of incident IDs.

```bash
python analysis/align.py
```

Outputs:

```text
emb_text.pkl
emb_transE.pkl
emb_graph.pkl
```

---

### Step 2: Correlation analysis

Compare the similarity structure of different embeddings using Pearson and Spearman correlation.

```bash
python analysis/correlation.py
```

---

### Step 3: Similarity heatmap

Visualize cosine similarity patterns within a single embedding.

```bash
python analysis/heatmap.py
```

Output:

```text
heatmap_graph.png
```

---

## Notes

* All embeddings are stored as:

  ```python
  Dict[int, np.ndarray]  # incident_id -> embedding
  ```
* Embedding generation and analysis are intentionally separated
* Random sampling is used for efficiency and visualization clarity
* The pipeline is fully modular and reproducible

---

## Summary

This repository provides a clean, modular pipeline for **constructing, aligning, and comparing incident embeddings** from multiple perspectives (semantic, structural, and relational), enabling both quantitative and qualitative analysis of their differences.


