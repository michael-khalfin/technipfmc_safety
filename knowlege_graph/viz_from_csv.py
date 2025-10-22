"""
Visualize knowledge graph from nodes/edges CSVs produced by process_csv.py

Features:
- Builds a NetworkX graph from `nodes.csv` (id,label,type) and `edges.csv` (src,rel,dst,...)
- Aggregates duplicate edges (same src,dst,rel) and stores a `weight` count
- Saves a static PNG via matplotlib and an interactive HTML via PyVis (if installed)

Usage (examples):
  python knowlege_graph/viz_from_csv.py \
      --nodes knowlege_graph/output/nodes.csv \
      --edges knowlege_graph/output/edges.csv \
      --top-n 300

If `--out-dir` is not provided, an `_viz` directory is created alongside the CSVs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize knowledge graph from nodes/edges CSVs.")
    p.add_argument("--nodes", type=Path, default=Path("knowlege_graph/output/nodes.csv"), help="Path to nodes.csv")
    p.add_argument("--edges", type=Path, default=Path("knowlege_graph/output/edges.csv"), help="Path to edges.csv")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory for visualization artifacts")
    p.add_argument("--top-n", type=int, default=250, help="Plot at most N highest-degree nodes (static/interactive)")
    p.add_argument("--label-top", type=int, default=40, help="Label only top-K nodes by degree in static plot")
    p.add_argument("--min-weight", type=float, default=1.0, help="Filter edges with weight below this value when plotting")
    p.add_argument("--seed", type=int, default=42, help="Random seed for layouts")
    p.add_argument("--directed", action="store_true", help="Build a directed graph instead of undirected")
    p.add_argument("--title", default="Knowledge Graph Visualization", help="Title for the static plot")
    return p.parse_args()


def ensure_pyvis() -> Optional[object]:
    try:
        from pyvis.network import Network  # type: ignore
        return Network
    except Exception:
        print("Warning: pyvis not installed. Skipping interactive HTML output. Install with: pip install pyvis")
        return None


def load_graph(nodes_csv: Path, edges_csv: Path, directed: bool = False) -> nx.Graph:
    if not nodes_csv.exists():
        raise FileNotFoundError(f"nodes CSV not found: {nodes_csv}")
    if not edges_csv.exists():
        raise FileNotFoundError(f"edges CSV not found: {edges_csv}")

    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)

    # Normalize and aggregate duplicate edges (src, dst, rel)
    for col in ("src", "dst", "rel"):
        if col not in edges_df.columns:
            raise ValueError(f"edges CSV missing required column: {col}")
    agg = (
        edges_df.groupby(["src", "dst", "rel"], dropna=False)
        .size()
        .reset_index(name="weight")
        .sort_values("weight", ascending=False)
    )

    # Choose graph type
    G: nx.Graph
    G = nx.DiGraph() if directed else nx.Graph()

    # Add nodes using IDs as keys, with attributes
    # Expected columns: id, label, type
    if "id" not in nodes_df.columns or "label" not in nodes_df.columns:
        raise ValueError("nodes CSV must have at least 'id' and 'label' columns")

    for _, r in nodes_df.iterrows():
        nid = str(r["id"])  # supports hashed IDs like 'node:abcd'
        label = str(r.get("label", nid))
        ntype = str(r.get("type", "entity"))
        G.add_node(nid, label=label, type=ntype)

    # Add edges with weight and label (relation)
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

        # For simple visualization, if an undirected edge already exists, update weight
        if not directed and G.has_edge(src, dst):
            existing_w = G[src][dst].get("weight", 1.0)
            G[src][dst]["weight"] = float(existing_w) + w
            # Prefer preserving a relation label if consistent; otherwise keep first
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
    # Use a spring layout with a k parameter scaling with graph size for better spacing
    try:
        import numpy as np  # type: ignore
        k = 1.0 / (np.sqrt(max(1, H.number_of_nodes()))) * 8.0
    except Exception:
        k = None
    pos = nx.spring_layout(H, seed=seed, k=k) if k else nx.spring_layout(H, seed=seed)

    # Compute degree for sizing and labeling
    degrees = dict(H.degree())
    # Label only top-K nodes by degree
    top_nodes_for_labels = set(sorted(degrees, key=degrees.get, reverse=True)[: max(0, label_top)])
    node_labels = {n: H.nodes[n].get("label", str(n)) for n in top_nodes_for_labels}
    # Size nodes by degree (with floor and scale)
    node_sizes = [max(40, min(400, 20 + 8 * degrees.get(n, 1))) for n in H.nodes]
    # Color groups by node type
    node_colors = [hash(H.nodes[n].get("type", "entity")) % 20 for n in H.nodes]

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
    Network = ensure_pyvis()
    if Network is None:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    net = Network(height="900px", width="100%", notebook=False, cdn_resources="in_line")
    net.force_atlas_2based()

    for node, attrs in G.nodes(data=True):
        title = attrs.get("label", str(node))
        ntype = attrs.get("type", "entity")
        hover = f"<b>{title}</b><br>Type: {ntype}<br>Degree: {G.degree(node)}"
        net.add_node(node, label=title, title=hover, group=ntype)

    for s, t, attrs in G.edges(data=True):
        w = float(attrs.get("weight", 1.0))
        rel = str(attrs.get("label", "related_to"))
        if w >= min_edge_weight:
            hover = f"<b>{rel}</b><br>Weight: {w:.0f}"
            net.add_edge(s, t, title=hover, value=w)

    html_path = out_dir / "graph_interactive.html"
    net.show_buttons(filter_=["physics"])
    # Generate HTML and write explicitly with UTF-8 to avoid Windows cp1252 encode errors
    try:
        html = net.generate_html(str(html_path))
    except Exception:
        # Fallback: some versions may not accept a name parameter
        html = net.generate_html()
    with html_path.open("w", encoding="utf-8") as f:
        f.write(html)
    return html_path


def main() -> None:
    args = parse_args()

    # Determine output directory
    if args.out_dir is None:
        # Put _viz next to provided nodes CSV
        out_dir = args.nodes.parent / "_viz"
    else:
        out_dir = args.out_dir

    print(f"Loading graph from:\n  nodes: {args.nodes}\n  edges: {args.edges}")
    G = load_graph(args.nodes, args.edges, directed=args.directed)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges before filtering")

    H = filter_top_n_by_degree(G, args.top_n)
    # Remove edges below weight threshold to reduce clutter
    if args.min_weight > 1.0:
        to_remove = [(u, v) for u, v, d in H.edges(data=True) if float(d.get("weight", 1.0)) < args.min_weight]
        if to_remove:
            H.remove_edges_from(to_remove)
    if H.number_of_nodes() != G.number_of_nodes():
        print(f"Filtered to top {args.top_n} nodes for plotting: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

    # Save static PNG
    png_path = plot_static(H, out_dir, args.title, args.seed, args.label_top)
    print(f"Saved static plot to: {png_path}")

    # Save interactive HTML (if pyvis installed)
    html_path = plot_interactive(H, out_dir)
    if html_path:
        print(f"Saved interactive HTML to: {html_path}")
    else:
        print("Interactive HTML skipped (pyvis not installed). Install with: pip install pyvis")


if __name__ == "__main__":
    main()
