import pandas as pd
import numpy as np
import json
import pickle
import re
import nltk

from nltk.stem import WordNetLemmatizer
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

"""
Incident-level embedding using TransE.

Pipeline:
1. Load raw incident triples
2. Clean and normalize relation text
3. Train TransE on cleaned triples
4. Extract entity embeddings
5. Map entity embeddings to incident embeddings
6. Save incident-level embeddings

Assumption:
Each incident is represented by its source entity in the knowledge graph.
"""


# =========================
# Configuration
# =========================
RAW_TRIPLE_PATH = "incident_triples.csv"
OUT_PATH = "event_transe_embedding.pkl"

EMBEDDING_DIM = 128
BATCH_SIZE = 256
LR = 1e-3
DEVICE = "cpu"


# =========================
# 1. Load raw triples
# =========================
df = pd.read_csv(RAW_TRIPLE_PATH)
# Expected columns: incident_id, source, target, description


# =========================
# 2. Clean relation text
# =========================

# Remove empty or missing relations
df = df[df["description"].notna()]
df = df[df["description"].str.strip() != ""]

def keep_last_line(text: str) -> str:
    """Keep only the last line of multi-line descriptions."""
    lines = str(text).splitlines()
    return lines[-1].strip()

df["description"] = df["description"].apply(keep_last_line)

# Remove relations containing digits
df = df[~df["description"].str.contains(r"\d")]

# Remove relations with too many words
df["relation_word_count"] = df["description"].apply(lambda x: len(x.split()))
df = df[df["relation_word_count"] < 5]
df.drop(columns=["relation_word_count"], inplace=True)


# =========================
# 3. Normalize relation text
# =========================
lemmatizer = WordNetLemmatizer()

def normalize_relation(rel: str) -> str:
    """
    Normalize relation text into a canonical form.
    """
    rel = rel.lower().strip()
    rel = rel.replace("_", " ")
    rel = re.sub(r"[^a-z\s]", "", rel)
    rel = re.sub(r"\s+", " ", rel).strip()

    words = [lemmatizer.lemmatize(w) for w in rel.split()]
    return "_".join(words)

df["relation"] = df["description"].apply(normalize_relation)
df = df[df["relation"].str.strip() != ""]


# =========================
# 4. Build triples for TransE
# =========================
triples = df[["source", "relation", "target"]].astype(str).to_numpy()
tf = TriplesFactory.from_labeled_triples(triples)

# Minimal split (TransE does not rely on test metrics here)
tf_train, tf_test = tf.split([0.9999, 0.0001])


# =========================
# 5. Train TransE
# =========================
result = pipeline(
    training=tf_train,
    testing=tf_test,
    model="TransE",
    model_kwargs=dict(
        embedding_dim=EMBEDDING_DIM,
        scoring_fct_norm=1,
    ),
    training_loop="sLCWA",
    training_kwargs=dict(
        batch_size=BATCH_SIZE,
    ),
    negative_sampler="basic",
    negative_sampler_kwargs=dict(
        num_negs_per_pos=1,
    ),
    optimizer="adam",
    optimizer_kwargs=dict(
        lr=LR,
    ),
    device=DEVICE,
)

model = result.model


# =========================
# 6. Extract entity embeddings
# =========================
entity_embeddings = (
    model.entity_representations[0](indices=None)
    .detach()
    .cpu()
    .numpy()
)

entity_to_id = {k: int(v) for k, v in tf.entity_to_id.items()}


# =========================
# 7. Map entity embeddings to incident embeddings
# =========================
event_embeddings = {}

for _, row in df.iterrows():
    incident_id = row["incident_id"]
    source_entity = row["source"]

    if source_entity not in entity_to_id:
        continue

    idx = entity_to_id[source_entity]
    event_embeddings[incident_id] = entity_embeddings[idx]


# =========================
# 8. Save incident-level embeddings
# =========================
with open(OUT_PATH, "wb") as f:
    pickle.dump(event_embeddings, f)

print(f"Saved {len(event_embeddings)} incident embeddings to {OUT_PATH}")
