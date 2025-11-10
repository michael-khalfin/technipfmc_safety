import pandas as pd
from pathlib import Path

out_dir = Path("graphRAG/output")
entities = pd.read_parquet(out_dir / "entities.parquet")
relationships = pd.read_parquet(out_dir / "relationships.parquet")


# Set the Mapping For Nodes
node_mappings = dict(
    id=lambda df: "node:" + df["id"],
    label=lambda df: df["title"],
    type=lambda df: df["type"].fillna("entity")
)
nodes = (
    entities.assign(**node_mappings)[["id", "label", "type"]]
)

id_lookup = entities.set_index("title")["id"]

# Set the Mapping for Edges
edge_mappings = dict(
    src_label=lambda df: df["source"],
    dst_label=lambda df: df["target"],
    src=lambda df: "node:" + df["source"].map(id_lookup),
    dst=lambda df: "node:" + df["target"].map(id_lookup),
    rel=lambda df: df["description"],
    row_id=lambda df: df["human_readable_id"],
)
edges = (
    relationships.assign(**edge_mappings)[["src", "rel", "dst", "src_label", "dst_label", "row_id"]]
)

print("Converted to CSV's")
evaluation_dir = Path("evaluation/KG3")
nodes.to_csv(evaluation_dir / "nodes.csv", index=False)
edges.to_csv( evaluation_dir / "edges.csv", index=False)
