import pandas as pd
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse
import os

def evaluate_semantic_similarity(nodes_path, edges_path, outdir="outputs", model_name="all-MiniLM-L6-v2", n_pairs=3000, seed=0):
    """Main function for semantic similarity evaluation, returns a dictionary containing all evaluation metrics"""
    # === Read Data ===
    os.makedirs(outdir, exist_ok=True)
    print(f"[Info] Reading data from {nodes_path} and {edges_path}")
    nodes = pd.read_csv(nodes_path)    # Must contain node_id,label
    edges = pd.read_csv(edges_path)    # Must contain src,dst

    G = nx.Graph()
    G.add_nodes_from(nodes["id"].tolist())
    G.add_edges_from(edges[["src","dst"]].itertuples(index=False, name=None))

    # === Text Embedding and Semantic Similarity Calculation ===
    print(f"[Info] Computing embeddings using {model_name}")
    model = SentenceTransformer(model_name)
    embeds = model.encode(nodes["label"].astype(str).tolist(), normalize_embeddings=True)
    emb_map = dict(zip(nodes["id"], embeds))

    def cosine(a,b): return float(np.dot(a,b))

    ids = nodes["id"].tolist()
    rng = random.Random(seed)
    pairs = []
    print(f"[Info] Sampling {n_pairs} node pairs...")
    for _ in tqdm(range(n_pairs)):
        u,v = rng.sample(ids, 2)
        try:
            dist = nx.shortest_path_length(G, u, v)
        except nx.NetworkXNoPath:
            continue
        sim = cosine(emb_map[u], emb_map[v])
        pairs.append((sim, dist))

    S = np.array([p[0] for p in pairs])
    D = np.array([p[1] for p in pairs])

    # === Calculate Spearman Correlation ===
    rho, pval = spearmanr(S, D)
    print(f"\n==== Semantic Similarity Results ====")
    print(f"Spearman r = {rho:.4f} (p={pval:.2e}), n={len(S)}")

    # === Visualization ===
    plt.figure(figsize=(7,5))
    plt.scatter(D, S, alpha=0.3, s=10)
    plt.xlabel("Graph distance")
    plt.ylabel("Text similarity (cosine)")
    plt.title(f"Text similarity vs Graph distance\nSpearman r={rho:.2f}")
    
    scatter_plot = os.path.join(outdir, "textsim_vs_distance_scatter.png")
    plt.savefig(scatter_plot, dpi=150)
    plt.close()

    #  === Box Plot: Grouped by Distance === 
    bins = sorted(set(D))
    data = [S[D==b] for b in bins if len(S[D==b])>5]
    plt.figure(figsize=(7,5))
    plt.boxplot(data, positions=[b for b in bins if len(S[D==b])>5], showfliers=False)
    plt.xlabel("Graph distance")
    plt.ylabel("Text similarity (cosine)")
    plt.title("Average similarity by graph distance")
    
    box_plot = os.path.join(outdir, "textsim_vs_distance_box.png")
    plt.savefig(box_plot, dpi=150)
    plt.close()

    print(f"[Plot] Saved plots to {outdir}/")

    return {
        "spearman_r": float(rho),
        "p_value": float(pval),
        "n_samples": int(len(S))
    }

def main():
    parser = argparse.ArgumentParser(description="Semantic Similarity vs Graph Distance Analysis")
    parser.add_argument("--nodes", type=str, required=True, help="Path to nodes.csv (must have: id,label)")
    parser.add_argument("--edges", type=str, required=True, help="Path to edges.csv (must have: src,dst)")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--n_pairs", type=int, default=3000, help="Number of node pairs to sample")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    evaluate_semantic_similarity(args.nodes, args.edges, args.model, args.n_pairs, args.seed)

if __name__ == "__main__":
    main()
