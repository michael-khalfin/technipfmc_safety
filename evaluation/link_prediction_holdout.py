import argparse
import random
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import math
from sklearn.metrics import average_precision_score, roc_auc_score
import os


def build_graph(nodes_path: str, edges_path: str, undirected: bool = True) -> nx.Graph:
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)

    # Expected columns: node_id / src,dst; please rename if different or rename after reading CSV
    node_ids = nodes["id"].tolist()
    edge_pairs = list(edges[["src", "dst"]].itertuples(index=False, name=None))

    G = nx.Graph() if undirected else nx.DiGraph()
    G.add_nodes_from(node_ids)
    G.add_edges_from(edge_pairs)
    return G


def sample_holdout(
    G_full: nx.Graph, test_ratio: float, rng: random.Random
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], nx.Graph]:
    """Sample positive test examples (real edges) and negative examples, and return the training graph G_train"""
    E = list(G_full.edges())
    n_test = max(1, int(test_ratio * len(E)))
    test_pos = rng.sample(E, n_test)

    G_train = G_full.copy()
    G_train.remove_edges_from(test_pos)

    # Negative samples: Sample the same amount from non-edges
    # Note: non_edges is a generator, convert to list first (for large graphs, can randomly sample a subset)
    non_edges = list(nx.non_edges(G_train))
    if len(non_edges) < n_test:
        n_test = len(non_edges)  # fallback
    test_neg = rng.sample(non_edges, n_test)

    return test_pos, test_neg, G_train


def _aa_scores(G: nx.Graph, pairs, eps=1e-9):
    scores = {}
    for (u, v) in pairs:
        s = 0.0
        if (u in G) and (v in G):
            for w in nx.common_neighbors(G, u, v):
                deg = G.degree(w)
                # log(1+eps) > 0, avoid division by zero; larger deg means contribution closer to original AA
                s += 1.0 / math.log(max(1.0 + eps, float(deg)))
        if u <= v:
            scores[(u, v)] = s
        else:
            scores[(v, u)] = s
    return scores

def _jc_scores(G: nx.Graph, pairs: List[Tuple[int, int]]) -> Dict[Tuple[int, int], float]:
    return {(u, v): s for u, v, s in nx.jaccard_coefficient(G, pairs)}


def _ra_scores(G: nx.Graph, pairs: List[Tuple[int, int]]) -> Dict[Tuple[int, int], float]:
    return {(u, v): s for u, v, s in nx.resource_allocation_index(G, pairs)}


def scores_to_arrays(
    score_dict: Dict[Tuple[int, int], float],
    test_pos: List[Tuple[int, int]],
    test_neg: List[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    # In NetworkX undirected graphs, (u,v) and (v,u) are equivalent; ensure key consistency
    def norm_edge(e):
        u, v = e
        return (u, v) if u <= v else (v, u)

    y, s = [], []
    for e in test_pos:
        y.append(1)
        s.append(score_dict.get(norm_edge(e), 0.0))
    for e in test_neg:
        y.append(0)
        s.append(score_dict.get(norm_edge(e), 0.0))
    return np.array(y), np.array(s)


def evaluate_scores(y: np.ndarray, s: np.ndarray, k: int = 100) -> dict:
    ap = average_precision_score(y, s)          # PR-AUC
    auc = roc_auc_score(y, s)                   # ROC-AUC

    # P@K / R@K
    K = min(k, len(y))
    idx = np.argsort(-s)[:K]
    y_top = y[idx]
    p_at_k = float(y_top.mean())
    r_at_k = float(y_top.sum() / (y.sum() + 1e-12))

    return {"PR-AUC": float(ap), "ROC-AUC": float(auc), "P@K": p_at_k, "R@K": r_at_k}


def evaluate_link_prediction(nodes_path, edges_path, outdir="outputs", test_ratio=0.1, seed=42, k=50, directed=False):
    """Main function for link prediction evaluation, returns a dictionary containing all evaluation metrics"""
    rng = random.Random(seed)

    # Create output directory
    os.makedirs(outdir, exist_ok=True)
    
    print(f"[Info] Building graph ...")
    G_full = build_graph(nodes_path, edges_path, undirected=not directed)
    print(f"[Info] #nodes={G_full.number_of_nodes()}, #edges={G_full.number_of_edges()}")

    print("[Info] Sampling test/negatives and constructing train graph ...")
    test_pos, test_neg, G_train = sample_holdout(G_full, test_ratio, rng)
    print(f"[Info] |test_pos|={len(test_pos)}, |test_neg|={len(test_neg)}, #edges_train={G_train.number_of_edges()}")

    # Candidate pairs for scoring = positive test samples + negative test samples
    candidate_pairs = test_pos + test_neg

    # Calculate scores for different methods
    results = {}
    
    print("[Info] Scoring with Adamic-Adar ...")
    aa = _aa_scores(G_train, candidate_pairs)
    y, s = scores_to_arrays(aa, test_pos, test_neg)
    aa_metrics = evaluate_scores(y, s, k=k)
    results["Adamic-Adar"] = aa_metrics

    print("[Info] Scoring with Jaccard ...")
    jc = _jc_scores(G_train, candidate_pairs)
    y, s = scores_to_arrays(jc, test_pos, test_neg)
    jc_metrics = evaluate_scores(y, s, k=k)
    results["Jaccard"] = jc_metrics

    print("[Info] Scoring with Resource Allocation ...")
    ra = _ra_scores(G_train, candidate_pairs)
    y, s = scores_to_arrays(ra, test_pos, test_neg)
    ra_metrics = evaluate_scores(y, s, k=k)
    results["ResourceAlloc"] = ra_metrics

    print("\n==== Link Prediction (Holdout) Results ====")
    
    def pretty(name, m):
        print(f"{name:>20s} | PR-AUC: {m['PR-AUC']:.4f} | ROC-AUC: {m['ROC-AUC']:.4f} | P@{k}: {m['P@K']:.4f} | R@{k}: {m['R@K']:.4f}")
    
    for name, metrics in results.items():
        pretty(name, metrics)

    #  === Draw Comparison Plots === 
    methods = ["Adamic-Adar", "Jaccard", "ResourceAlloc"]
    aucs = [aa_metrics["ROC-AUC"], jc_metrics["ROC-AUC"], ra_metrics["ROC-AUC"]]
    prs  = [aa_metrics["PR-AUC"], jc_metrics["PR-AUC"], ra_metrics["PR-AUC"]]
    pats = [aa_metrics["P@K"], jc_metrics["P@K"], ra_metrics["P@K"]]
    rats = [aa_metrics["R@K"], jc_metrics["R@K"], ra_metrics["R@K"]]

    x = range(len(methods))
    plt.figure(figsize=(8,5))
    plt.bar(x, aucs, width=0.35, label="ROC-AUC")
    plt.bar([i+0.35 for i in x], prs, width=0.35, label="PR-AUC")
    plt.xticks([i+0.17 for i in x], methods)
    plt.ylabel("Score")
    plt.title("Link-Prediction AUC Comparison")
    plt.legend()
    plt.tight_layout()
    
    auc_plot = os.path.join(outdir, "linkpred_auc_compare.png")
    plt.savefig(auc_plot, dpi=150)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.bar(x, pats, width=0.35, label=f"P@{k}")
    plt.bar([i+0.35 for i in x], rats, width=0.35, label=f"R@{k}")
    plt.xticks([i+0.17 for i in x], methods)
    plt.ylabel("Score")
    plt.title("Precision / Recall at K Comparison")
    plt.legend()
    plt.tight_layout()
    
    pr_plot = os.path.join(outdir, "linkpred_pr_compare.png")
    plt.savefig(pr_plot, dpi=150)
    plt.close()

    print(f"[Plot] Saved plots to {outdir}/")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Link Prediction Holdout Evaluation")
    parser.add_argument("--nodes", type=str, required=True, help="Path to nodes.csv (need column: node_id)")
    parser.add_argument("--edges", type=str, required=True, help="Path to edges.csv (need columns: src,dst)")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Fraction of true edges for test (default 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--k", type=int, default=50, help="K for P@K / R@K")
    parser.add_argument("--directed", action="store_true", help="Use directed graph (default undirected)")
    args = parser.parse_args()

    evaluate_link_prediction(args.nodes, args.edges, args.test_ratio, args.seed, args.k, args.directed)

if __name__ == "__main__":
    main()
    