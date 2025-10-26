"""
Visualize knowledge graph from nodes/edges CSVs produced by process_csv.py

Features:
- Builds a NetworkX graph from `nodes.csv` (id,label,type) and `edges.csv` (src,rel,dst,...)
- Aggregates duplicate edges (same src,dst,rel) and stores a `weight` count
- Saves a static PNG via matplotlib and an interactive HTML via PyVis (if installed)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network

# File Configs
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
NODES_CSV = OUTPUT_DIR / "nodes.csv"
EDGES_CSV = OUTPUT_DIR / "edges.csv"
VIZ_DIR = OUTPUT_DIR / "_viz"

# General Configs
SEED = 123
TOP_N = 250
LABEL_TOP = 40
MIN_WEIGHT = 1.0
DIRECTED = False
PLOT_TITLE = "Knowledge Graph Visualization"

def load_graph(nodes_csv: Path, edges_csv: Path, directed: bool = False) -> nx.Graph:
    if not nodes_csv.exists():
        raise FileNotFoundError(f"nodes CSV not found: {nodes_csv}")
    if not edges_csv.exists():
        raise FileNotFoundError(f"edges CSV not found: {edges_csv}")

    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)

    # Normalize and aggregate duplicate edges
    for col in ("src", "dst", "rel"):
        if col not in edges_df.columns:
            raise ValueError(f"edges CSV missing required column: {col}")
    agg = (
        edges_df.groupby(["src", "dst", "rel"], dropna=False)
        .size()
        .reset_index(name="weight")
        .sort_values("weight", ascending=False)
    )


    G: nx.Graph
    G = nx.DiGraph() if directed else nx.Graph()

    # Add nodes using IDs as keys, with attributes
    for _, r in nodes_df.iterrows():
        nid = str(r["id"])  
        label = str(r.get("label", nid))
        ntype = str(r.get("type", "entity"))
        G.add_node(nid, label=label, type=ntype)

    # Add edges with weight and label 
    for _, r in agg.iterrows():
        src = str(r["src"])
        dst = str(r["dst"])
        rel = str(r["rel"]) if pd.notna(r["rel"]) else "related_to"
        w = float(r["weight"]) if pd.notna(r["weight"]) else 1.0

        # Ensure endpoints exist (guard against edges referencing filtered/missing nodes)
        if src not in G:
            G.add_node(src, label=src, type="entity")
        if dst not in G:
            G.add_node(dst, label=dst, type="entity")

        #if an undirected edge already exists, update weight
        if not directed and G.has_edge(src, dst):
            existing_w = G[src][dst].get("weight", 1.0)
            G[src][dst]["weight"] = float(existing_w) + w
        else:
            G.add_edge(src, dst, weight=w, label=rel)

    return G


def filter_top_n_by_degree(G: nx.Graph, top_n: Optional[int]) -> nx.Graph:
    if top_n is None or G.number_of_nodes() <= top_n:
        return G
    top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[: top_n]
    keep = {n for n, _ in top_nodes}
    return G.subgraph(keep).copy()


def plot_static(G: nx.Graph, out_dir: Path, title: str, seed: int, label_top: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    H = G
    k = 1.0 / (np.sqrt(max(1, H.number_of_nodes()))) * 8.0
    pos = nx.spring_layout(H, seed=seed, k=k) if k else nx.spring_layout(H, seed=seed)

    degrees = dict(H.degree())
    # Label only top-K nodes by degree
    top_nodes_for_labels = set(sorted(degrees, key=degrees.get, reverse=True)[: max(0, label_top)])
    node_labels = {n: H.nodes[n].get("label", str(n)) for n in top_nodes_for_labels}
    node_sizes = [max(40, min(400, 20 + 8 * degrees.get(n, 1))) for n in H.nodes]
    node_colors = [hash(H.nodes[n].get("type", "entity")) % 20 for n in H.nodes] # same cuz we only process entity? 

    plt.figure(figsize=(20, 20))
    nx.draw_networkx_nodes(H, pos, node_size=node_sizes, alpha=0.9, node_color=node_colors, cmap="tab20")
    nx.draw_networkx_edges(H, pos, alpha=0.2)
    if node_labels:
        nx.draw_networkx_labels(H, pos, labels=node_labels, font_size=7, font_color="black")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    png_path = out_dir / "graph_static.png"
    plt.savefig(png_path, dpi=150)
    plt.close()
    return png_path


def plot_interactive(G: nx.Graph, out_dir: Path, min_edge_weight: float = 0.0) -> Optional[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    net = Network(height="900px", width="100%", notebook=False, cdn_resources="in_line")
    net.force_atlas_2based()

    for node, attrs in G.nodes(data=True):
        title = attrs.get("label", str(node))
        ntype = attrs.get("type", "entity")
        hover = f"Title: {title}, Type: {ntype}, Degree: {G.degree(node)}"
        net.add_node(node, label=title, title=hover, group=ntype)

    for s, t, attrs in G.edges(data=True):
        w = float(attrs.get("weight", 1.0))
        rel = str(attrs.get("label", "related_to"))
        if w >= min_edge_weight:
            hover = f"Relation: {rel}, Weight: {w:.0f}"
            net.add_edge(s, t, title=hover, value=w)

    html_path = out_dir / "graph_interactive.html"
    net.show_buttons(filter_=["physics"])
    try:
        html = net.generate_html(str(html_path))
    except Exception:
        html = net.generate_html()
    with html_path.open("w", encoding="utf-8") as f:
        f.write(html)
    return html_path


def main() -> None:
    out_dir = VIZ_DIR or (NODES_CSV.parent / "_viz")

    print(f"Loading graph from:\n  nodes: {NODES_CSV}\n  edges: {EDGES_CSV}")
    G = load_graph(NODES_CSV, EDGES_CSV, directed=DIRECTED)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges before filtering")

    H = filter_top_n_by_degree(G, TOP_N)
    # Remove edges below weight threshold to reduce clutter
    if MIN_WEIGHT > 1.0:
        to_remove = [(u, v) for u, v, d in H.edges(data=True) if float(d.get("weight", 1.0)) < MIN_WEIGHT]
        if to_remove:
            H.remove_edges_from(to_remove)
    if H.number_of_nodes() != G.number_of_nodes():
        print(f"Filtered to top {TOP_N} nodes for plotting: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

    # Save static PNG
    png_path = plot_static(H, out_dir, PLOT_TITLE, SEED, LABEL_TOP)
    print(f"Saved static plot to: {png_path}")

    # Save interactive HTML
    html_path = plot_interactive(H, out_dir, min_edge_weight=MIN_WEIGHT)
    if html_path:
        print(f"Saved interactive HTML to: {html_path}")
    else:
        print("Interactive HTML skipped")


if __name__ == "__main__":
    main()
