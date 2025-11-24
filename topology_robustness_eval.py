"""Topology robustness evaluation for event/knowledge graphs.

Computes a compact set of topology metrics (connectivity, clustering,
community structure, degree/assortativity, k-core) and writes a JSON
summary. Designed to be lightweight and consistent with other evaluation
scripts in this repo.

Usage examples:
  python topology_robustness_eval.py KG1
  python topology_robustness_eval.py --nodes KG1/nodes.csv --edges KG1/edges.csv --out KG1/output/topology_metrics.json

Dependencies: pandas, networkx
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional

import os
import pandas as pd
import networkx as nx
from networkx.algorithms import community as nx_comm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_nodes(nodes_csv: str) -> pd.DataFrame:
    df = pd.read_csv(nodes_csv)
    return df


def read_edges(edges_csv: str) -> pd.DataFrame:
    df = pd.read_csv(edges_csv)
    return df


def detect_edge_cols(df: pd.DataFrame):
    candidates = [
        ("source", "target"),
        ("src", "dst"),
        ("u", "v"),
        ("head", "tail"),
        ("from", "to"),
        ("node1", "node2"),
    ]
    cols = [c.lower() for c in df.columns]
    for a, b in candidates:
        if a in cols and b in cols:
            # return original-case column names
            real_a = [c for c in df.columns if c.lower() == a][0]
            real_b = [c for c in df.columns if c.lower() == b][0]
            return real_a, real_b
    # fallback: use first two columns
    return df.columns[0], df.columns[1]


def detect_node_id_col(df: pd.DataFrame):
    cols = [c.lower() for c in df.columns]
    for candidate in ("id", "node", "node_id", "nid", "name"):
        if candidate in cols:
            return [c for c in df.columns if c.lower() == candidate][0]
    return df.columns[0]


def build_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
    # Build directed and undirected graphs. We keep both for different metrics.
    Gd = nx.DiGraph()
    nid = detect_node_id_col(nodes_df)
    for _, r in nodes_df.iterrows():
        try:
            Gd.add_node(r[nid], **r.drop(labels=[nid]).to_dict())
        except Exception:
            # if node id can't be used as key, cast to str
            Gd.add_node(str(r[nid]), **r.drop(labels=[nid]).to_dict())

    a_col, b_col = detect_edge_cols(edges_df)
    for _, r in edges_df.iterrows():
        a = r[a_col]
        b = r[b_col]
        # ignore self-loops? keep for counts
        try:
            Gd.add_edge(a, b)
        except Exception:
            Gd.add_edge(str(a), str(b))

    G = Gd.to_undirected(reciprocal=False)
    return Gd, G


def giant_component_stats(G: nx.Graph) -> Dict:
    n = G.number_of_nodes()
    components = list(nx.connected_components(G)) if n > 0 else []
    num_comp = len(components)
    sizes = sorted([len(c) for c in components], reverse=True)
    largest = sizes[0] if sizes else 0
    giant_ratio = largest / n if n > 0 else 0.0
    avg_comp = float(sum(sizes) / num_comp) if num_comp > 0 else 0.0
    med_comp = float(sorted(sizes)[len(sizes) // 2]) if sizes else 0.0
    return {
        "n_nodes": n,
        "n_components": num_comp,
        "giant_component_size": largest,
        "giant_component_ratio": giant_ratio,
        "avg_component_size": avg_comp,
        "median_component_size": med_comp,
    }


def path_stats(G: nx.Graph) -> Dict:
    # compute on largest connected component
    if G.number_of_nodes() == 0:
        return {"avg_shortest_path": None, "diameter": None}
    comps = list(nx.connected_components(G))
    largest = max(comps, key=len)
    if len(largest) <= 1:
        return {"avg_shortest_path": None, "diameter": None}
    Gsub = G.subgraph(largest)
    
    # Skip computation if component is too large (expensive O(n^2) operation)
    # For large graphs, computing all-pairs shortest paths is prohibitively expensive
    if len(Gsub) > 5000:
        print(f"  [path_stats] Skipping diameter/avg_shortest_path for large component (n={len(Gsub)})")
        return {"avg_shortest_path": None, "diameter": None}
    
    try:
        avg = nx.average_shortest_path_length(Gsub)
    except Exception as e:
        print(f"  [path_stats] Failed to compute avg_shortest_path: {e}")
        avg = None
    try:
        diam = nx.diameter(Gsub)
    except Exception as e:
        print(f"  [path_stats] Failed to compute diameter: {e}")
        diam = None
    return {"avg_shortest_path": avg, "diameter": diam}


def clustering_stats(G: nx.Graph) -> Dict:
    if G.number_of_nodes() == 0:
        return {
            "avg_clustering": None,
            "global_transitivity": None,
        }
    avg_c = nx.average_clustering(G)
    trans = nx.transitivity(G)
    return {"avg_clustering": avg_c, "global_transitivity": trans}


def community_stats(G: nx.Graph, top_n_communities: int = 10) -> Dict:
    # Using greedy modularity communities (no extra deps)
    if G.number_of_nodes() == 0:
        return {"n_communities": 0, "modularity": None}
    try:
        communities = list(nx_comm.greedy_modularity_communities(G))
        modularity = nx_comm.modularity(G, communities)
    except Exception:
        communities = [set(G.nodes())]
        modularity = None

    # map node->community id
    node2comm = {}
    for i, com in enumerate(communities):
        for v in com:
            node2comm[v] = i

    # participation coefficient and within-module z-score
    deg = dict(G.degree())
    comm_nodes = defaultdict(list)
    for n, cid in node2comm.items():
        comm_nodes[cid].append(n)

    # compute within-community degrees
    within_deg = {n: 0 for n in G.nodes()}
    for n in G.nodes():
        for nbr in G[n]:
            if node2comm.get(n) == node2comm.get(nbr):
                within_deg[n] += 1

    comm_stats = {}
    for cid, nodes in comm_nodes.items():
        vals = [within_deg[n] for n in nodes]
        mean = float(sum(vals) / len(vals)) if vals else 0.0
        std = float((sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5) if vals else 0.0
        comm_stats[cid] = {"size": len(nodes), "mean_within_deg": mean, "std_within_deg": std}

    P = {}
    Z = {}
    for n in G.nodes():
        k = deg.get(n, 0)
        if k == 0:
            P[n] = 0.0
            Z[n] = 0.0
            continue
        # participation coefficient
        sum_frac_sq = 0.0
        # aggregate neighbors per community
        per_comm = defaultdict(int)
        for nbr in G[n]:
            per_comm[node2comm.get(nbr, -1)] += 1
        for ccount in per_comm.values():
            frac = ccount / k
            sum_frac_sq += frac * frac
        P[n] = 1.0 - sum_frac_sq
        # within-module z-score
        cid = node2comm.get(n)
        mu = comm_stats[cid]["mean_within_deg"] if cid in comm_stats else 0.0
        sigma = comm_stats[cid]["std_within_deg"] if cid in comm_stats else 0.0
        kz = (within_deg[n] - mu) / sigma if sigma > 0 else 0.0
        Z[n] = kz

    # summary counts
    bridges = sum(1 for v in P.values() if v > 0.5)
    cores = sum(1 for v in Z.values() if v > 1.0)

    top_comms = sorted([(i, len(c)) for i, c in enumerate(communities)], key=lambda x: x[1], reverse=True)[:top_n_communities]
    community_sizes = [len(c) for c in communities]

    return {
        "n_communities": len(communities),
        "modularity": modularity,
        "top_communities": [{"id": int(i), "size": int(s)} for i, s in top_comms],
        "community_sizes": community_sizes,
        "n_bridges_P_gt_0.5": int(bridges),
        "n_cores_z_gt_1.0": int(cores),
    }


def save_plots(G: nx.Graph, community_sizes: List[int], out_dir: str, prefix: str = "topology") -> None:
    """Save degree distribution and community size histogram into out_dir."""
    # Degree histogram
    degrees = [d for _, d in G.degree()]
    if degrees:
        plt.figure(figsize=(6, 4))
        plt.hist(degrees, bins=50, color="#4C72B0")
        plt.xlabel("Degree")
        plt.ylabel("Count")
        plt.title("Degree Distribution")
        p1 = os.path.join(out_dir, f"{prefix}_degree_hist.png")
        plt.tight_layout()
        plt.savefig(p1, dpi=150)
        plt.close()

        # log-log degree frequency
        deg_counts = Counter(degrees)
        degs = sorted(deg_counts.keys())
        freqs = [deg_counts[d] for d in degs]
        plt.figure(figsize=(6, 4))
        plt.loglog(degs, freqs, marker="o", linestyle="none")
        plt.xlabel("Degree (log)")
        plt.ylabel("Frequency (log)")
        plt.title("Degree Frequency (log-log)")
        p2 = os.path.join(out_dir, f"{prefix}_degree_loglog.png")
        plt.tight_layout()
        plt.savefig(p2, dpi=150)
        plt.close()

    # Community size histogram
    if community_sizes:
        sizes = sorted(community_sizes, reverse=True)
        plt.figure(figsize=(6, 4))
        plt.bar(range(len(sizes)), sizes, color="#55A868")
        plt.xlabel("Community (ranked)")
        plt.ylabel("Size")
        plt.title("Community Sizes (ranked)")
        p3 = os.path.join(out_dir, f"{prefix}_community_sizes.png")
        plt.tight_layout()
        plt.savefig(p3, dpi=150)
        plt.close()



def degree_and_assortativity(Gd: nx.DiGraph, G: nx.Graph, nodes_df: pd.DataFrame) -> Dict:
    degs = [d for _, d in G.degree()]
    if not degs:
        return {}
    deg_ctr = Counter(degs)
    mean_deg = float(sum(degs) / len(degs))
    median_deg = float(sorted(degs)[len(degs) // 2])
    max_deg = max(degs)
    frac_deg1 = sum(1 for d in degs if d == 1) / len(degs)

    # degree assortativity (numeric)
    try:
        # Patch: manually compute to avoid zero denominator
        from networkx.algorithms.assortativity import degree_pearsonr
        r, t, s = degree_pearsonr(G)
        # If denominator (1-s) is zero, add epsilon
        denom = 1 - s
        if abs(denom) < 1e-8:
            denom += 1e-8
        assort = (t - s) / denom
    except Exception:
        try:
            assort = nx.degree_assortativity_coefficient(G)
        except Exception:
            assort = None

    # categorical assortativity if node type exists
    type_assort = None
    type_col_candidates = [c for c in nodes_df.columns if c.lower() in ("type", "entity_type", "label_type")]
    if type_col_candidates:
        tcol = type_col_candidates[0]
        try:
            # ensure attribute exists on graph nodes
            for _, r in nodes_df.iterrows():
                nid = detect_node_id_col(nodes_df)
                if r[nid] in G:
                    G.nodes[r[nid]][tcol] = r[tcol]
            type_assort = nx.attribute_assortativity_coefficient(G, tcol)
        except Exception:
            type_assort = None

    # k-core
    try:
        core_nums = nx.core_number(G)
        max_core = max(core_nums.values()) if core_nums else 0
        nodes_in_max_core = [n for n, k in core_nums.items() if k == max_core]
    except Exception:
        max_core = None
        nodes_in_max_core = []

    return {
        "mean_degree": mean_deg,
        "median_degree": median_deg,
        "max_degree": int(max_deg),
        "frac_degree_1": float(frac_deg1),
        "degree_assortativity": assort,
        "type_assortativity": type_assort,
        "max_k_core": int(max_core) if max_core is not None else None,
        "n_nodes_in_max_k_core": len(nodes_in_max_core),
    }


def summarize_metrics(metrics: Dict) -> None:
    # concise human-readable summary
    print("Topology Robustness Summary:")
    print(f" - Nodes: {metrics['connectivity']['n_nodes']}")
    print(f" - Giant component ratio: {metrics['connectivity']['giant_component_ratio']:.3f}")
    asp = metrics['paths']
    if asp['avg_shortest_path'] is not None:
        print(f" - Avg shortest path (giant): {asp['avg_shortest_path']:.3f}")
    if asp['diameter'] is not None:
        print(f" - Diameter (giant): {asp['diameter']}")
    cl = metrics['clustering']
    print(f" - Avg clustering: {cl['avg_clustering']:.4f}, Transitivity: {cl['global_transitivity']:.4f}")
    comm = metrics['community']
    print(f" - Communities: {comm['n_communities']}, Modularity: {comm['modularity']}")
    deg = metrics['degree']
    print(f" - Mean/median/max degree: {deg.get('mean_degree')} / {deg.get('median_degree')} / {deg.get('max_degree')}")


def run(nodes_csv: str, edges_csv: str, outpath: Optional[str] = None):
    nodes_df = read_nodes(nodes_csv)
    edges_df = read_edges(edges_csv)
    Gd, G = build_graph(nodes_df, edges_df)

    metrics = {}
    metrics['connectivity'] = giant_component_stats(G)
    metrics['paths'] = path_stats(G)
    metrics['clustering'] = clustering_stats(G)
    metrics['community'] = community_stats(G)
    metrics['degree'] = degree_and_assortativity(Gd, G, nodes_df)

    summarize_metrics(metrics)

    # determine output folder for plots / json
    if outpath:
        out_dir = os.path.dirname(outpath) or os.path.dirname(nodes_csv)
    else:
        out_dir = os.path.join(os.path.dirname(nodes_csv), "output")
    os.makedirs(out_dir, exist_ok=True)

    # save plots: degree distribution and community size histogram
    try:
        community_sizes = metrics['community'].get('community_sizes', [])
        save_plots(G, community_sizes, out_dir, prefix="topology")
    except Exception as e:
        print(f"Warning: failed to save plots: {e}")

    # write JSON results
    json_out = outpath or os.path.join(out_dir, "topology_metrics.json")
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Wrote JSON results to {json_out}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", help="KG folder path containing nodes.csv and edges.csv (or provide --nodes and --edges)")
    parser.add_argument("--nodes", help="Path to nodes.csv")
    parser.add_argument("--edges", help="Path to edges.csv")
    parser.add_argument("--out", help="Path to output JSON file")
    args = parser.parse_args()

    nodes = args.nodes
    edges = args.edges
    if args.input and not (nodes and edges):
        import os

        folder = args.input
        nodes_candidate = os.path.join(folder, "nodes.csv")
        edges_candidate = os.path.join(folder, "edges.csv")
        if os.path.exists(nodes_candidate) and os.path.exists(edges_candidate):
            nodes = nodes or nodes_candidate
            edges = edges or edges_candidate

    if not nodes or not edges:
        parser.error("must provide --nodes and --edges or a folder containing nodes.csv and edges.csv")

    run(nodes, edges, args.out)


if __name__ == "__main__":
    main()
