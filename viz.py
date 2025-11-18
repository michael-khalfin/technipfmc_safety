"""
GraphRAG Parquet → Graph Visualization (NetworkX + PyVis)

This script loads GraphRAG parquet outputs (entities, relationships, etc.),
builds a graph, and generates an interactive HTML file for visualization.

The script creates both static and interactive visualizations of the knowledge graph
generated from safety incident data, allowing for exploration of relationships
and patterns in the data.
"""

# --- Config ---
# IMPORTANT: Manually set this path to the specific, timestamped output directory you want to visualize.
# Example: PIPELINE_OUTPUT_DIR = "./graphRAG/output/20251012-164458"
PIPELINE_OUTPUT_DIR = "./graphRAG/output"  # <--- EDIT THIS PATH

# Filter out weak edges (a value between 0.0 and 1.0)
MIN_EDGE_WEIGHT = 0.0
# For large graphs, limit the plot to the top N nodes by degree. Set to None to plot all.
TOP_N_NODES = 250
# Random seed for consistent layouts
SEED = 42
# --- End of Config ---

import os
from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import socket # To get the hostname for instructions

# Check for PyVis for interactive plotting
try:
    from pyvis.network import Network
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False
    print("Warning: pyvis is not installed. Interactive HTML plot will not be generated.")
    print("You can install it with: pip install pyvis")

# --- 1. Find and Load Data ---
OUTPUT_PATH = Path(PIPELINE_OUTPUT_DIR)

if not OUTPUT_PATH.exists() or "EDIT THIS PATH" in str(OUTPUT_PATH):
    print("="*60)
    print("!! PLEASE EDIT THE SCRIPT !!")
    print(f"Error: The directory '{PIPELINE_OUTPUT_DIR}' does not exist or has not been set.")
    print("Please edit the 'PIPELINE_OUTPUT_DIR' variable at the top of this script")
    print("to point to the correct timestamped output folder.")
    print("="*60)
    exit()

print(f"Visualizing data from: {OUTPUT_PATH}")
VIZ_DIR = OUTPUT_PATH / "_viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)


isFiltered = True
if isFiltered:
    entities_pq = OUTPUT_PATH / "entities_filtered.parquet"
    rels_pq = OUTPUT_PATH / "relationships_filtered.parquet"
else:
    entities_pq = OUTPUT_PATH / "entities.parquet"
    rels_pq = OUTPUT_PATH / "relationships.parquet"

print("\nChecking for parquet files:")
if not entities_pq.exists() or not rels_pq.exists():
    print(f"Error: Could not find 'entities.parquet' or 'relationships.parquet' in {OUTPUT_PATH}")
    exit()
else:
    print(f"  {entities_pq}: Exists")
    print(f"  {rels_pq}: Exists")

ents = pd.read_parquet(entities_pq)
rels = pd.read_parquet(rels_pq)

print("\nCounts:")
print(f"  Entities: {len(ents)}")
print(f"  Relationships: {len(rels)}")

# --- 2. Build NetworkX Graph ---
print("\nBuilding graph...")

# Infer column names from common candidates
SRC_COL = next((c for c in ["source", "source_id", "src_id"] if c in rels.columns), None)
TGT_COL = next((c for c in ["target", "target_id", "dst_id"] if c in rels.columns), None)
WEIGHT_COL = next((c for c in ["weight", "edge_weight"] if c in rels.columns), None)
LABEL_COL = next((c for c in ["description", "relation", "label"] if c in rels.columns), None)
ENT_KEY_COL = next((c for c in ["title", "name", "human_readable_id"] if c in ents.columns), "id")

if not SRC_COL or not TGT_COL:
    raise ValueError("Could not automatically determine source/target columns in relationships.parquet.")

print(f"Using columns -> ENTITY_KEY:{ENT_KEY_COL}, SRC:{SRC_COL}, TGT:{TGT_COL}, WEIGHT:{WEIGHT_COL}")

# Normalize keys for consistent matching (e.g., 'Forklift' vs 'FORKLIFT')
def normalize_key(x):
    return str(x).strip().upper()

ents["_node_key"] = ents[ENT_KEY_COL].apply(normalize_key)
rels = rels[rels[SRC_COL].notna() & rels[TGT_COL].notna()].copy()
rels["_src_key"] = rels[SRC_COL].apply(normalize_key)
rels["_tgt_key"] = rels[TGT_COL].apply(normalize_key)

# Create a unique set of entities for nodes
ents_unique = ents.drop_duplicates(subset=["_node_key"]).copy()
node_keys = set(ents_unique["_node_key"])

G = nx.Graph()

# Add nodes to the graph
for _, row in ents_unique.iterrows():
    node_key = row["_node_key"]
    # Store all other attributes
    attrs = row.drop(labels=["_node_key"]).to_dict()
    G.add_node(node_key, **attrs)

# Add edges to the graph
for _, row in rels.iterrows():
    src_key, tgt_key = row["_src_key"], row["_tgt_key"]
    
    edge_attrs = {}
    if WEIGHT_COL and row[WEIGHT_COL] is not None:
        edge_attrs["weight"] = float(row[WEIGHT_COL])
    if LABEL_COL and row[LABEL_COL] is not None:
        edge_attrs["label"] = str(row[LABEL_COL])

    G.add_edge(src_key, tgt_key, **edge_attrs)
    
print(f"Graph built successfully: Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}")

# --- 3. Create a Static Visualization (Matplotlib) ---
print("\nGenerating static plot...")
H = G.copy()

# Filter subgraph if TOP_N_NODES is set
if TOP_N_NODES is not None and H.number_of_nodes() > TOP_N_NODES:
    top_nodes = sorted(H.degree, key=lambda x: x[1], reverse=True)[:TOP_N_NODES]
    keep_nodes = {n for n, _ in top_nodes}
    H = H.subgraph(keep_nodes).copy()
    print(f"Filtered to top {TOP_N_NODES} nodes for plotting.")

print(f"Plotting subgraph: Nodes={H.number_of_nodes()}, Edges={H.number_of_edges()}")
pos = nx.spring_layout(H, seed=SEED, k=0.8)
node_labels = {n: H.nodes[n].get("title", str(n)) for n in H.nodes()}

plt.figure(figsize=(20, 20))
nx.draw_networkx_nodes(H, pos, node_size=50, alpha=0.9)
nx.draw_networkx_edges(H, pos, alpha=0.3)
nx.draw_networkx_labels(H, pos, labels=node_labels, font_size=8, font_color="darkred")

plt.axis("off")
plt.title(f"GraphRAG Visualization (Top {TOP_N_NODES} nodes)", size=16)
plt.tight_layout()
static_plot_path = VIZ_DIR / "graph_static.png"
plt.savefig(static_plot_path, dpi=150)
print(f"Saved static plot to {static_plot_path}")

# --- 4. Create an Interactive Visualization (PyVis) ---
html_path = VIZ_DIR / "graph_interactive.html"
if HAS_PYVIS:
    print("\nGenerating interactive HTML plot...")
    net = Network(height="900px", width="100%", notebook=False, cdn_resources="in_line")
    net.force_atlas_2based()

    for node, attrs in H.nodes(data=True):
        title = attrs.get("title", str(node))
        node_type = attrs.get("type", "Unknown")
        hover_text = f"<b>{title}</b><br>Type: {node_type}<br>Degree: {H.degree(node)}"
        net.add_node(node, label=title, title=hover_text, group=node_type)

    for source, target, attrs in H.edges(data=True):
        weight = attrs.get("weight", 0)
        edge_label = attrs.get("label", "")
        # Truncate long labels for display
        display_label = edge_label[:30] + "..." if len(edge_label) > 30 else edge_label
        hover_text = f"<b>Relationship:</b><br>{edge_label}<br>Weight: {weight:.2f}"
        if weight >= MIN_EDGE_WEIGHT:
            net.add_edge(source, target, label=display_label, title=hover_text, value=weight)

    net.show_buttons(filter_=['physics'])
    net.write_html(str(html_path), notebook=False, open_browser=False)
    print(f"Saved interactive HTML to {html_path}")

print("\n--- Visualization complete! ---")

# --- 5. Print Instructions to Download and View ---
print("\nINSTRUCTIONS TO VIEW YOUR INTERACTIVE GRAPH:")
print("-" * 60)
print("The interactive graph has been saved as an HTML file. To view it, follow these steps:")
print("\n1. Open a NEW local terminal on your own computer (not this SSH session).")
print("\n2. Use the following 'scp' command to download the file to your local 'Downloads' folder.")
print("   (You will be prompted for your cluster password)")

# Get the full path of the generated HTML file for the user
full_html_path = html_path.resolve()
hostname = socket.gethostname()

print(f"\n   scp your_username@{hostname}:{full_html_path} ~/Downloads/\n")
print("   (Replace 'your_username' with your actual username for the cluster)")

print("\n3. After the download is complete, find the 'graph_interactive.html' file in your")
print("   local 'Downloads' folder and open it with your web browser (e.g., Chrome, Firefox).")
print("-" * 60)