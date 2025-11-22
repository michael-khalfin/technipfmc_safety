#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bootstrap Consistency (BC) for sparse KGs

Usage:
  python bootstrap_consistency_eval.py /path/to/KG_DIR \
      --k_sem 10 --k_graph 10 --bootstrap 20 \
      --graph_neigh khop --khop 2 \
      --edge_bootstrap sample --sample_ratio 0.9 \
      --seed 42 --out results_bc.json

Inputs:
  KG_DIR/nodes.csv: id,label,type
  KG_DIR/edges.csv: src,rel,dst,(others ignored)

Outputs:
  JSON with:
    - global mean/std Jaccard across bootstraps
    - per-type mean/std
    - per-bootstrap diagnostics
"""

import argparse
import json
import os
import sys
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import networkx as nx

# --- optional deps handling (sentence-transformers is preferred) ---
def get_embeddings(labels, model_name="sentence-transformers/all-MiniLM-L6-v2", use_tfidf_fallback=True):
    """
    Return np.ndarray [N, d], preserving order of input labels (list of str).
    If sentence-transformers unavailable, optionally fallback to TF-IDF.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        X = model.encode(labels, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        return X
    except Exception as e:
        if not use_tfidf_fallback:
            raise
        # TF-IDF fallback
        print(f"[BC] sentence-transformers not available -> fallback to TF-IDF ({e})", file=sys.stderr)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
        vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        X = vec.fit_transform(labels)
        X = normalize(X)  # L2 normalize for cosine consistency
        return X.toarray()

def knn_neighbors(X, k, metric="cosine"):
    """
    Returns list of neighbor ID sets (length N), each is top-k indices excluding self.
    Uses sklearn NearestNeighbors to support cosine metric.
    """
    from sklearn.neighbors import NearestNeighbors
    N = X.shape[0]
    k_eff = min(k+1, N)  # +1 to include self then remove
    nbrs = NearestNeighbors(n_neighbors=k_eff, metric=metric, n_jobs=-1)
    nbrs.fit(X)
    dists, inds = nbrs.kneighbors(X, return_distance=True)
    neigh_sets = []
    for i in range(N):
        idxs = [j for j in inds[i] if j != i]
        neigh_sets.append(set(idxs[:k]))
    return neigh_sets

# --- graph helpers ---
def load_graph(nodes_csv, edges_csv):
    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)
    # normalize column names
    nodes.columns = [c.strip().lower() for c in nodes.columns]
    edges.columns = [c.strip().lower() for c in edges.columns]
    # required columns
    assert {"id","label","type"}.issubset(nodes.columns), "nodes.csv needs columns: id,label,type"
    assert {"src","dst"}.issubset(edges.columns), "edges.csv needs columns: src,dst"
    # build undirected simple graph for neighborhood computations
    G = nx.Graph()
    for _, r in nodes.iterrows():
        G.add_node(r["id"], label=str(r["label"]), type=str(r["type"]))
    # Add edges; ignore multi-edges and keep simple undirected connectivity
    for _, r in edges.iterrows():
        u, v = r["src"], r["dst"]
        if pd.isna(u) or pd.isna(v): 
            continue
        if u in G and v in G and u != v:
            G.add_edge(u, v)
    # ensure node ordering stable
    id_list = list(nodes["id"].tolist())
    return G, nodes, edges, id_list

def graph_neighbors_khop(G, id2idx, idx2id, k_graph=10, khop=2):
    """
    For each node, return a set of up to k_graph neighbors collected from nodes within <=khop hops,
    prioritizing by shortest-path length then by degree (tie-breaker stable).
    """
    neigh_sets = []
    for i, nid in enumerate(idx2id):
        if nid not in G:
            neigh_sets.append(set())
            continue
        # BFS up to khop
        dists = nx.single_source_shortest_path_length(G, nid, cutoff=khop)
        # exclude self
        cand = [n for n, d in dists.items() if n != nid]
        # sort by (distance asc, degree desc, id stable)
        cand.sort(key=lambda x: (dists[x], -G.degree(x), x))
        chosen = [id2idx[c] for c in cand[:k_graph] if c in id2idx]
        neigh_sets.append(set(chosen))
    return neigh_sets

def graph_neighbors_ppr_topk(G, id2idx, idx2id, k_graph=10, alpha=0.85):
    """
    Personalized PageRank top-k neighbors per node (exclude self). 
    NOTE: O(N * PR) can be slow on very large graphs. Use khop for scale.
    """
    neigh_sets = []
    for i, nid in enumerate(idx2id):
        if nid not in G or G.degree(nid) == 0:
            neigh_sets.append(set())
            continue
        pr = nx.pagerank(G, alpha=alpha, personalization={nid:1.0})
        # sort by score desc, drop self
        cand = [n for n,_s in sorted(pr.items(), key=lambda kv: kv[1], reverse=True) if n != nid]
        chosen = [id2idx[c] for c in cand[:k_graph] if c in id2idx]
        neigh_sets.append(set(chosen))
    return neigh_sets

def jaccard_per_node(sem_sets, graph_sets):
    assert len(sem_sets) == len(graph_sets)
    N = len(sem_sets)
    vals = np.zeros(N, dtype=float)
    for i in range(N):
        a, b = sem_sets[i], graph_sets[i]
        if not a and not b:
            vals[i] = 0.0  # define empty-empty as 0 (conservative)
        else:
            inter = len(a & b)
            union = len(a | b) if (a or b) else 1
            vals[i] = inter / union
    return vals

def edge_bootstrap_resample(G_base, sample_ratio=0.9, with_replacement=True, seed=None):
    """
    Create a bootstrap graph by resampling edges.
    - with_replacement=True: sample m edges with replacement, then make simple graph
    - with_replacement=False: random subset of edges (dropout)
    Keep the same node set as base.
    """
    rng = np.random.default_rng(seed)
    edges = list(G_base.edges())
    m = len(edges)
    if m == 0:
        return G_base.copy()
    if with_replacement:
        k = max(1, int(round(m * sample_ratio)))
        sampled = [edges[idx] for idx in rng.integers(0, m, size=k)]
    else:
        k = max(1, int(round(m * sample_ratio)))
        idxs = rng.choice(m, size=k, replace=False)
        sampled = [edges[i] for i in idxs]
    H = nx.Graph()
    H.add_nodes_from(G_base.nodes(data=True))
    H.add_edges_from(sampled)
    return H

def summarize_by_type(jaccard_vals, nodes_df, idx2id):
    """
    Return dict: type -> {mean, std, count}
    """
    id2type = {r["id"]: r["type"] for _, r in nodes_df.iterrows()}
    per_type = defaultdict(list)
    for i, nid in enumerate(idx2id):
        t = id2type.get(nid, "NA")
        per_type[t].append(jaccard_vals[i])
    out = {}
    for t, arr in per_type.items():
        arr = np.array(arr, dtype=float)
        out[t] = {
            "count": int(arr.size),
            "mean": float(arr.mean()) if arr.size else 0.0,
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kg_dir", type=str, help="Directory containing nodes.csv and edges.csv")
    ap.add_argument("--k_sem", type=int, default=10, help="k for semantic kNN")
    ap.add_argument("--k_graph", type=int, default=10, help="k for graph neighborhood")
    ap.add_argument("--graph_neigh", type=str, default="khop", choices=["khop","ppr"], help="Graph neighborhood method")
    ap.add_argument("--khop", type=int, default=2, help="hop distance when graph_neigh=khop")
    ap.add_argument("--alpha", type=float, default=0.85, help="PPR alpha when graph_neigh=ppr")
    ap.add_argument("--bootstrap", type=int, default=20, help="number of bootstrap replicates")
    ap.add_argument("--edge_bootstrap", type=str, default="sample", choices=["sample","dropout"], 
                    help="'sample' = with replacement; 'dropout' = without replacement")
    ap.add_argument("--sample_ratio", type=float, default=0.9, help="edge sample ratio per bootstrap")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--no_tfidf_fallback", action="store_true", help="disable TF-IDF fallback")
    ap.add_argument("--out", type=str, default="results_bc.json")
    args = ap.parse_args()

    nodes_csv = os.path.join(args.kg_dir, "nodes.csv")
    edges_csv = os.path.join(args.kg_dir, "edges.csv")
    assert os.path.exists(nodes_csv) and os.path.exists(edges_csv), "nodes.csv/edges.csv not found"

    # 1) Load graph
    G_base, nodes_df, edges_df, id_list = load_graph(nodes_csv, edges_csv)
    print(f"[BC] Nodes={G_base.number_of_nodes()} Edges={G_base.number_of_edges()}")

    # Index maps (stable)
    idx2id = id_list
    id2idx = {nid: i for i, nid in enumerate(idx2id)}
    # Ensure all nodes in G_base
    for nid in idx2id:
        if nid not in G_base:
            G_base.add_node(nid, label=str(nodes_df.loc[nodes_df['id']==nid, 'label'].values[0]), 
                                 type=str(nodes_df.loc[nodes_df['id']==nid, 'type'].values[0]))

    # 2) Semantic embeddings + kNN (fixed across bootstraps)
    labels_in_order = [str(G_base.nodes[nid].get("label", "")) for nid in idx2id]
    X = get_embeddings(labels_in_order, model_name=args.model_name, use_tfidf_fallback=not args.no_tfidf_fallback)
    sem_sets = knn_neighbors(X, k=args.k_sem, metric="cosine")

    # 3) Bootstraps
    rng = np.random.default_rng(args.seed)
    with_replacement = (args.edge_bootstrap == "sample")
    B = args.bootstrap
    boot_stats = []
    all_node_j = []  # store per-bootstrap mean only; full per-node omitted for size

    for b in range(B):
        seed_b = int(rng.integers(0, 1_000_000))
        H = edge_bootstrap_resample(G_base, sample_ratio=args.sample_ratio, 
                                    with_replacement=with_replacement, seed=seed_b)
        # choose graph neighborhood
        if args.graph_neigh == "khop":
            graph_sets = graph_neighbors_khop(H, id2idx, idx2id, k_graph=args.k_graph, khop=args.khop)
        else:
            graph_sets = graph_neighbors_ppr_topk(H, id2idx, idx2id, k_graph=args.k_graph, alpha=args.alpha)

        # Jaccard per node
        jac = jaccard_per_node(sem_sets, graph_sets)
        # summarize
        global_mean = float(jac.mean()) if jac.size else 0.0
        global_std  = float(jac.std(ddof=1)) if jac.size > 1 else 0.0
        per_type = summarize_by_type(jac, nodes_df, idx2id)

        boot_stats.append({
            "bootstrap_id": b,
            "global_mean": global_mean,
            "global_std": global_std,
            "per_type": per_type,
            "edges_bootstrapped": H.number_of_edges()
        })
        all_node_j.append(global_mean)

        print(f"[BC] b={b:02d} | edges={H.number_of_edges()} | mean J={global_mean:.4f}")

    # 4) Aggregate across bootstraps
    gm = np.array([s["global_mean"] for s in boot_stats], dtype=float)
    global_summary = {
        "B": B,
        "edge_bootstrap": args.edge_bootstrap,
        "sample_ratio": args.sample_ratio,
        "graph_neigh": args.graph_neigh,
        "k_sem": args.k_sem,
        "k_graph": args.k_graph,
        "khop": args.khop if args.graph_neigh=="khop" else None,
        "alpha": args.alpha if args.graph_neigh=="ppr" else None,
        "mean_of_means": float(gm.mean()) if gm.size else 0.0,
        "std_of_means": float(gm.std(ddof=1)) if gm.size > 1 else 0.0,
        "min_mean": float(gm.min()) if gm.size else 0.0,
        "max_mean": float(gm.max()) if gm.size else 0.0
    }

    # per-type aggregate across bootstraps: average the per-type means
    # (only for types that appear; missing types are skipped)
    per_type_all = defaultdict(list)
    for s in boot_stats:
        for t, d in s["per_type"].items():
            per_type_all[t].append(d["mean"])
    per_type_summary = {}
    for t, arr in per_type_all.items():
        arr = np.array(arr, dtype=float)
        per_type_summary[t] = {
            "B_covered": int(arr.size),
            "mean_of_means": float(arr.mean()) if arr.size else 0.0,
            "std_of_means": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min_mean": float(arr.min()) if arr.size else 0.0,
            "max_mean": float(arr.max()) if arr.size else 0.0
        }

    results = {
        "global_summary": global_summary,
        "per_type_summary": per_type_summary,
        "bootstraps": boot_stats
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[BC] Done. Saved -> {args.out}")

if __name__ == "__main__":
    main()
