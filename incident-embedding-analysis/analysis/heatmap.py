import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances
from scipy.cluster.hierarchy import linkage, leaves_list


def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    # Load embedding (example: graph embedding)
    emb = load_pkl("embed_heatmap/emb_graph.pkl")

    # Load metadata for sampling (severity, etc.)
    df = pd.read_csv("cleaned_data_without_unnecessary_features.csv")

    ID_COL = "RECORD_NO_LOSS_POTENTIAL"
    SEVERITY_COL = "SEVERITY_VALUE"

    df = df.dropna(subset=[ID_COL, SEVERITY_COL])
    df[ID_COL] = df[ID_COL].astype(int)
    df[SEVERITY_COL] = df[SEVERITY_COL].astype(int)
    df = df.drop_duplicates(subset=[ID_COL])

    # Keep only incidents that have embeddings
    df = df[df[ID_COL].isin(emb.keys())]

    # Random sampling
    df_sampled = df.sample(n=min(len(df), 300), random_state=42)
    id_list = df_sampled[ID_COL].tolist()

    # Build embedding matrix
    X = np.vstack([emb[i] for i in id_list])
    X = normalize(X)

    # Cosine similarity
    sim = 1 - cosine_distances(X)

    # Hierarchical clustering
    Z = linkage(sim, method="average")
    order = leaves_list(Z)
    sim = sim[order][:, order]

    # Mask upper triangle
    mask = np.triu(np.ones_like(sim), k=1).astype(bool)
    sim_masked = np.ma.array(sim, mask=mask)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(sim_masked, cmap="viridis")
    plt.colorbar(label="Cosine Similarity")
    plt.xlabel("Event Index")
    plt.ylabel("Event Index")
    plt.title("Cosine Similarity Heatmap (Graph Embedding)")
    plt.savefig(
        "heatmap_graph.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0
    )
