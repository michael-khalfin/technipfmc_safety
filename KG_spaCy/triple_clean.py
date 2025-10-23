#!/usr/bin/env python3
"""
Clean and normalize (subject, relation, object) triples from a CSV file using spaCy:
- input CSV columns: record_no, subject, relation, object (first column ignored)
- entities: lowercase, remove punctuation, merge duplicates
- relations: lemmatize verbs, unify format
- filter out triples where any entity > 4 words
- output: nodes.csv, edges.csv, and summary stats
"""

import re
import spacy
import pandas as pd
from tqdm import tqdm
from collections import Counter

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# -----------------------------
# Utility functions
# -----------------------------
def normalize_entity(text: str) -> str:
    """Lowercase, strip, and remove punctuation"""
    text = text.lower().strip()
    text = re.sub(r"[\"'.,;:!?(){}\[\]]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def normalize_relation(rel: str) -> str:
    """Lemmatize verbs and unify relation format"""
    doc = nlp(rel.lower().strip())
    lemmas = [t.lemma_ for t in doc if not t.is_stop]
    cleaned = "_".join(lemmas)
    cleaned = re.sub(r"[^a-z_]", "", cleaned)
    return cleaned

def entity_word_count(text: str) -> int:
    return len(text.split())

# -----------------------------
# Main cleaning process
# -----------------------------
def clean_triples_from_csv(csv_path, max_entity_words=4):
    """
    Read triples from CSV (columns: record_no, subject, relation, object)
    Return cleaned nodes and edges DataFrames
    """
    df = pd.read_csv(csv_path)  # auto-handle tab or comma
    required_cols = ["subject", "relation", "object"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    triples = list(zip(df["subject"], df["relation"], df["object"]))

    entity_map = {}
    next_id = 1
    edges = []
    filtered_out = 0

    for h, r, t in tqdm(triples, desc="Cleaning triples"):
        if pd.isna(h) or pd.isna(r) or pd.isna(t):
            continue

        h_norm = normalize_entity(str(h))
        t_norm = normalize_entity(str(t))
        r_norm = normalize_relation(str(r))

        # Filter overly long entities
        if entity_word_count(h_norm) > max_entity_words or entity_word_count(t_norm) > max_entity_words:
            filtered_out += 1
            continue

        for ent in [h_norm, t_norm]:
            if ent not in entity_map:
                entity_map[ent] = next_id
                next_id += 1

        edges.append({
            "source": entity_map[h_norm],
            "relation": r_norm,
            "target": entity_map[t_norm],
        })

    df_nodes = pd.DataFrame([{"id": v, "entity": k} for k, v in entity_map.items()])
    df_edges = pd.DataFrame(edges)

    print(f"{len(triples)} triples read from CSV")
    print(f"{filtered_out} triples removed (entity > {max_entity_words} words)")
    print(f"{len(df_nodes)} unique entities, {len(df_edges)} edges")

    # Relation statistics
    rel_counts = Counter(df_edges["relation"])
    print("\n🔹 Top relations:")
    for rel, cnt in rel_counts.most_common(10):
        print(f"   {rel:20s} {cnt}")

    # Degree statistics
    degree = Counter(df_edges["source"]) + Counter(df_edges["target"])
    print(f"\nAvg node degree: {sum(degree.values()) / len(df_nodes):.2f}")

    # Save outputs
    df_nodes.to_csv("nodes.csv", index=False)
    df_edges.to_csv("edges.csv", index=False)
    print("\nCleaned graph saved as nodes.csv and edges.csv")

    return df_nodes, df_edges

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    csv_path = "knowledge_graph_edges.csv"  # <-- replace with your actual file
    clean_triples_from_csv(csv_path)
