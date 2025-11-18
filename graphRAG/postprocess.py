"""
Post-process GraphRAG parquet outputs to remove metadata-heavy nodes/edges.

Usage:
    python graphRAG/postprocess.py 
"""

import argparse
from pathlib import Path
from typing import Optional
import pandas as pd

ALLOWED_TYPES = {
    "INCIDENT_TYPE",
    "INJURY_TYPE",
    "BODY_PART",
    "EQUIPMENT",
    "LOCATION",
    "ORGANIZATION",
    "DATE",
}

DROP_RELATIONS = {
    "SOURCE_SYSTEM",
    "META_FACTS",
    "REPORTED_DATE",
    "GENERAL_BUSINESS_UNIT",
    "SPECIFIC_BUSINESS_UNIT",
}

DROP_NODES = {
    "WORK RELATED CASE",
    "NOT A SIGNIFICANT INCIDENT FAILURE POTENTIAL",
    "NA - N/A - NON PROJECT SPECIFIC",
    "NOT APPLICABLE"
}

TITLE_COL = "title"
TYPE_COL = "type"
DEGREE = "degree"


class PostProcess:
    @staticmethod
    def _normalize(series: pd.Series) -> pd.Series:
        return series.fillna("").astype(str).str.strip().str.upper()

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.entities = pd.read_parquet(output_dir / "entities.parquet")
        self.relationships = pd.read_parquet(output_dir / "relationships.parquet")
        print(f"Number of Entities: {len(self.entities)}")
        print(f"Number of Relationship: {len(self.relationships)}")

        self.source_col = next((c for c in ["source", "source_id", "src_id"] if c in self.relationships.columns), None)
        self.target_col = next((c for c in ["target", "target_id", "dst_id"] if c in self.relationships.columns), None)
        self.relation_label_col = next((c for c in ["description", "relation", "label"] if c in self.relationships.columns), None)

        if not self.source_col or not self.target_col:
            raise ValueError("Could not locate source/target columns in relationships parquet.")
        
    
    def drop_k_degree_nodes(self, k_degree : Optional[int] = None):
        if not k_degree:
            k_degree = int(len(self.entities) * (0.9)) # Hit's > 90% of nodes or more 
        
        print(f"Removing Nodes with k-degree: {k_degree}")
        mask = ~(self.entities[DEGREE] >= k_degree)
        self.entities = self.entities[mask].copy()
        self._prune_edges()
    
    def drop_isolated_nodes(self):
        before = len(self.entities)
        mask = ~(self.entities[DEGREE] == 0)
        self.entities = self.entities[mask].copy()
        print(f"Dropped {before - len(self.entities)} ISOLATED nodes.")
        self._prune_edges()

    def drop_meta(self):
        before = len(self.entities)
        mask = ~self._normalize(self.entities[TITLE_COL]).str.startswith("META[")
        self.entities = self.entities[mask].copy()
        print(f"Dropped {before - len(self.entities)} META nodes.")
        self._prune_edges()

    def keep_only_allowed(self, allowed_types: set[str]):
        before = len(self.entities)
        mask = self._normalize(self.entities[TYPE_COL]).isin(allowed_types)
        self.entities = self.entities[mask].copy()
        print(f"Dropped {before - len(self.entities)} nodes outside allowed types.")
        self._prune_edges()

    def drop_relations_by_label(self, labels: set[str]):
        if not self.relation_label_col:
            print("No relation label column detected; skipping drop-relations step.")
            return
        before = len(self.relationships)
        mask = ~self._normalize(self.relationships[self.relation_label_col]).isin(labels)
        self.relationships = self.relationships[mask].copy()
        print(f"Dropped {before - len(self.relationships)} relationships by label.")

    def drop_node(self, node_name: str):
        norm_name = node_name.strip().upper()
        mask = self._normalize(self.entities[TITLE_COL]) == norm_name
        if not mask.any():
            raise ValueError(f"Node '{node_name}' does not exist in entities.parquet.")
        self.entities = self.entities[~mask].copy()
        print(f"Dropped node '{node_name}'.")
        self._prune_edges()

    def _prune_edges(self):
        keep_nodes = set(self._normalize(self.entities[TITLE_COL]))
        before = len(self.relationships)
        src_norm = self._normalize(self.relationships[self.source_col])
        tgt_norm = self._normalize(self.relationships[self.target_col])
        mask = src_norm.isin(keep_nodes) & tgt_norm.isin(keep_nodes)
        self.relationships = self.relationships[mask].copy()
        print(f"Dropped {before - len(self.relationships)} relationships referencing removed nodes.\n")

    def save(self, suffix: str):
        print(f"[SAVING] Number of Entities: {len(self.entities)}")
        print(f"[SAVING] Number of Relationship: {len(self.relationships)}")
        entities_path = self.output_dir / f"entities_{suffix}.parquet"
        rels_path = self.output_dir / f"relationships_{suffix}.parquet"
        self.entities.to_parquet(entities_path, index=False)
        self.relationships.to_parquet(rels_path, index=False)
        print(f"Saved filtered entities to {entities_path}")
        print(f"Saved filtered relationships to {rels_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Post-process GraphRAG parquet files.")
    parser.add_argument(
        "--dest-suffix",
        default="filtered",
        help="Suffix appended to the saved parquet files.",
    )
    parser.add_argument(
        "-A", "--allowed",
        default=False,
        action="store_true",
        help="Entity types to keep (overrides default schema filter).",
    )
    parser.add_argument(
        "-D", "--drop-labels",
        default=False,
        action="store_true",
        help="Relationship labels to drop (overrides default metadata labels).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path("graphRAG/output")
    processor = PostProcess(output_path)

    processor.drop_meta()
    processor.drop_k_degree_nodes()

    for node_name in DROP_NODES:
        processor.drop_node(node_name)

    if args.allowed:
        processor.keep_only_allowed(ALLOWED_TYPES)

    if args.drop_labels:
        processor.drop_relations_by_label(DROP_RELATIONS)

    processor.drop_isolated_nodes()
    processor.save(args.dest_suffix)

if __name__ == "__main__":
    main()
