import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr


def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_matrix(emb_map: dict, ids: list) -> np.ndarray:
    """Stack embeddings into a matrix following the given ID order."""
    return np.vstack([emb_map[i] for i in ids])


def flatten_lower_triangle(mat: np.ndarray) -> np.ndarray:
    """Flatten the lower triangular part of a matrix (excluding diagonal)."""
    idx = np.tril_indices(mat.shape[0], k=-1)
    return mat[idx]


def compute_correlation(x: np.ndarray, y: np.ndarray):
    """Compute Pearson and Spearman correlation."""
    return pearsonr(x, y)[0], spearmanr(x, y)[0]


if __name__ == "__main__":
    # Load aligned embeddings
    text_emb = load_pkl("embed_heatmap/emb_text.pkl")
    graph_emb = load_pkl("embed_heatmap/emb_graph.pkl")
    transe_emb = load_pkl("embed_heatmap/emb_transE.pkl")

    # Common incident IDs
    common_ids = list(
        set(text_emb.keys()) &
        set(graph_emb.keys()) &
        set(transe_emb.keys())
    )

    # Optional random sampling to reduce computation
    np.random.seed(42)
    sample_size = min(500, len(common_ids))
    sample_ids = np.random.choice(common_ids, size=sample_size, replace=False)

    # Build embedding matrices
    X_text = build_matrix(text_emb, sample_ids)
    X_graph = build_matrix(graph_emb, sample_ids)
    X_transe = build_matrix(transe_emb, sample_ids)

    # Pairwise cosine similarity
    S_text = cosine_similarity(X_text)
    S_graph = cosine_similarity(X_graph)
    S_transe = cosine_similarity(X_transe)

    # Flatten similarities
    v_text = flatten_lower_triangle(S_text)
    v_graph = flatten_lower_triangle(S_graph)
    v_transe = flatten_lower_triangle(S_transe)

    # Correlation results
    pairs = [
        ("Text", "Graph", compute_correlation(v_text, v_graph)),
        ("Text", "TransE", compute_correlation(v_text, v_transe)),
        ("Graph", "TransE", compute_correlation(v_graph, v_transe)),
    ]

    print("=== Embedding Correlation (Pearson / Spearman) ===")
    for a, b, (rp, rs) in pairs:
        print(f"{a:5s} vs {b:6s}: Pearson={rp:.4f}, Spearman={rs:.4f}")
