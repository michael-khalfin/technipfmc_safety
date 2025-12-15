import os
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

"""
Incident-level text embedding.

Pipeline:
1. Load cleaned incident text descriptions
2. Encode text using a sentence transformer
3. Reduce embedding dimensionality with PCA
4. Map embeddings to incident IDs
5. Save incident-level embeddings

Assumption:
Each row corresponds to one incident, and row order is consistent
between the text column and the incident ID column.
"""

# =========================
# Configuration
# =========================
DATA_PATH = "cleaned_description_translated.csv"
OUT_PATH = "event_text_embedding.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64
PCA_DIM = 128

# Optional: local cache for transformer models
os.environ["TRANSFORMERS_CACHE"] = "./models"


# =========================
# 1. Load data
# =========================
df = pd.read_csv(DATA_PATH)
# Expected columns:
# - "text": cleaned incident description
# - "RECORD_NO_LOSS_POTENTIAL": incident ID

texts = df["text"].astype(str).tolist()
incident_ids = df["RECORD_NO_LOSS_POTENTIAL"].astype(int).tolist()


# =========================
# 2. Encode text
# =========================
model = SentenceTransformer(MODEL_NAME)

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True
)
# shape: (num_incidents, original_dim=384)


# =========================
# 3. Dimensionality reduction (PCA)
# =========================
pca = PCA(n_components=PCA_DIM)
embeddings_reduced = pca.fit_transform(embeddings)
# shape: (num_incidents, PCA_DIM)


# =========================
# 4. Map incident ID to embedding
# =========================
event_embeddings = {}

for incident_id, emb in zip(incident_ids, embeddings_reduced):
    event_embeddings[incident_id] = emb


# =========================
# 5. Save incident-level embeddings
# =========================
with open(OUT_PATH, "wb") as f:
    pickle.dump(event_embeddings, f)

print(f"Saved {len(event_embeddings)} incident embeddings to {OUT_PATH}")
