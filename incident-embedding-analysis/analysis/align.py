import pickle
from typing import Dict, List


def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def align_embeddings(
    emb_maps: List[Dict[int, object]]
) -> List[Dict[int, object]]:
    """
    Align multiple embedding dictionaries by keeping only
    the intersection of incident IDs and enforcing the same order.

    Args:
        emb_maps: list of {incident_id -> embedding}

    Returns:
        list of aligned embedding dictionaries
    """
    common_ids = set(emb_maps[0].keys())
    for emb in emb_maps[1:]:
        common_ids &= set(emb.keys())

    common_ids = sorted(common_ids)
    print(f"Number of common incident IDs: {len(common_ids)}")

    aligned = []
    for emb in emb_maps:
        aligned.append({i: emb[i] for i in common_ids})

    return aligned


if __name__ == "__main__":
    # Load original embeddings
    text_emb = load_pkl("embedding_compare/id2text_emb.pkl")
    transe_emb = load_pkl("embedding_compare/id2temb.pkl")
    graph_emb = load_pkl("embedding_compare/id2gemb.pkl")

    # Align
    text_aligned, transe_aligned, graph_aligned = align_embeddings(
        [text_emb, transe_emb, graph_emb]
    )

    # Save aligned embeddings
    save_pkl(text_aligned, "embedding_compare/emb_text.pkl")
    save_pkl(transe_aligned, "embedding_compare/emb_transE.pkl")
    save_pkl(graph_aligned, "embedding_compare/emb_graph.pkl")

    print("Aligned embeddings saved.")
