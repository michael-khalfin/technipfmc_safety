import argparse
import os
import re
import json
import math
import pandas as pd
import numpy as np
from tqdm import tqdm

import sklearn
from packaging import version
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt


def normalize_name(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return ' '.join(s.split())


def entropy_from_counts(counts: pd.Series) -> float:
    p = counts / max(1, counts.sum())
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if len(p) else 0.0


def compute_embeddings(labels, model_name: str):
    print(f"[Info] Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    print("[Info] Encoding labels...")
    emb = model.encode(list(map(str, labels)), normalize_embeddings=True, show_progress_bar=True)
    return emb


def cluster_with_version_compat(embeddings: np.ndarray, distance_threshold: float):
    """
    Compatibility for sklearn 1.3 (affinity) and 1.4+ (metric)
    """
    skl_ver = version.parse(sklearn.__version__)
    print(f"[Info] scikit-learn version: {sklearn.__version__}")

    if skl_ver >= version.parse("1.4"):
        cluster = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="euclidean",
            linkage="average"
        )
    else:
        cluster = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            affinity="euclidean",
            linkage="average"
        )
    print(f"[Info] Clustering with distance_threshold={distance_threshold} ...")
    labels = cluster.fit_predict(embeddings)
    return labels


def compute_redundancy_consistency(nodes_df: pd.DataFrame):
    """
    Calculate:
      - Unique Entity Rate (UER)
      - Type Consistency Rate (TCR)
      - Name Ambiguity Entropy (NAE)
    Required columns in nodes_df: id, label, type, cluster_id
    """
    # Size of each cluster
    cluster_sizes = nodes_df.groupby("cluster_id")["id"].nunique()
    duplicate_clusters = (cluster_sizes > 1).sum()
    unique_entity_rate = 1 - duplicate_clusters / max(1, len(cluster_sizes))

    # Type consistency rate
    type_per_cluster = nodes_df.groupby("cluster_id")["type"].nunique()
    type_consistency_rate = (type_per_cluster == 1).mean()

    # Name ambiguity (based on normalized names)
    tmp = nodes_df.copy()
    tmp["norm"] = tmp["label"].apply(normalize_name)
    name_groups = tmp.groupby("norm")["type"]
    name_ambiguity_entropy = name_groups.apply(lambda s: entropy_from_counts(s.value_counts())).mean()

    return {
        "unique_entity_rate": float(unique_entity_rate),
        "type_consistency_rate": float(type_consistency_rate),
        "name_ambiguity_entropy": float(name_ambiguity_entropy),
        "n_clusters": int(len(cluster_sizes)),
        "n_nodes": int(len(nodes_df)),
    }


def export_cluster_tables(nodes_df: pd.DataFrame, outdir: str, top_k: int = 20, min_cluster_size: int = 2):
    """
    Export:
      - clusters.csv: Each node's cluster_id, cluster size, and type distribution per cluster
      - top_ambiguous_clusters.csv: Top k clusters sorted by type entropy
    """
    os.makedirs(outdir, exist_ok=True)

    # 簇大小
    cluster_sizes = nodes_df.groupby("cluster_id")["id"].count().rename("cluster_size")
    df = nodes_df.merge(cluster_sizes, on="cluster_id", how="left")

    # Type distribution per cluster (stored in JSON for readability)
    type_counts = (nodes_df.groupby(["cluster_id", "type"])["id"]
                   .count().rename("count").reset_index())
    type_map = (type_counts.groupby("cluster_id")
                .apply(lambda g: dict(zip(g["type"], g["count"]))).rename("type_counts"))

    df = df.merge(type_map, on="cluster_id", how="left")

    out_clusters = os.path.join(outdir, "clusters.csv")
    df.to_csv(out_clusters, index=False, encoding="utf-8")
    print(f"[Save] {out_clusters}")

    # Calculate type entropy for each cluster, filter clusters with size ≥min_cluster_size
    def cluster_entropy(cid):
        counts = type_counts[type_counts["cluster_id"] == cid]["count"]
        return entropy_from_counts(counts)

    cids = df["cluster_id"].unique()
    rows = []
    for cid in cids:
        size = int(cluster_sizes.loc[cid])
        if size >= min_cluster_size:
            e = cluster_entropy(cid)
            rows.append({"cluster_id": int(cid), "cluster_size": size, "type_entropy": float(e),
                        "type_counts": type_map.loc[cid] if cid in type_map.index else {}})

    amb = pd.DataFrame(rows).sort_values(["type_entropy", "cluster_size"], ascending=[False, False]).head(top_k)
    out_amb = os.path.join(outdir, "top_ambiguous_clusters.csv")
    amb.to_csv(out_amb, index=False, encoding="utf-8")
    print(f"[Save] {out_amb}")

    return df, amb


def visualize(outdir: str, nodes_df: pd.DataFrame, amb_df: pd.DataFrame):
    os.makedirs(outdir, exist_ok=True)

    # 1) Cluster size histogram
    plt.figure()
    cluster_sizes = nodes_df.groupby("cluster_id")["id"].count()
    plt.hist(cluster_sizes.values, bins=50)
    plt.title("Cluster Size Histogram")
    plt.xlabel("Cluster Size")
    plt.ylabel("Frequency")
    fig1 = os.path.join(outdir, "cluster_size_hist.png")
    plt.savefig(fig1, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[Plot] {fig1}")

    # 2) Type count per cluster histogram
    plt.figure()
    type_nums = nodes_df.groupby("cluster_id")["type"].nunique()
    plt.hist(type_nums.values, bins=20)
    plt.title("Type-Count per Cluster Histogram")
    plt.xlabel("#Distinct Types in Cluster")
    plt.ylabel("Frequency")
    fig2 = os.path.join(outdir, "typecount_hist.png")
    plt.savefig(fig2, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[Plot] {fig2}")

    # 3) Bar charts of type distribution for top "most ambiguous" clusters (sorted by entropy)
    #   Draw type distribution for each cluster, filename: amb_cluster_{cid}.png
    for _, row in amb_df.iterrows():
        cid = int(row["cluster_id"])
        tc = row["type_counts"]
        if not isinstance(tc, dict):
            try:
                tc = json.loads(tc)
            except Exception:
                continue
        if not tc:
            continue

        # Single cluster bar chart
        plt.figure()
        xs = list(tc.keys())
        ys = [tc[k] for k in xs]
        plt.bar(range(len(xs)), ys)
        plt.xticks(range(len(xs)), xs, rotation=45, ha="right")
        plt.title(f"Type Distribution (cluster {cid})")
        plt.xlabel("Type")
        plt.ylabel("Count")
        fname = os.path.join(outdir, f"amb_cluster_{cid}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"[Plot] {fname}")


def evaluate_entity_consistency(nodes_path, threshold=0.35, model_name="all-MiniLM-L6-v2", 
                            outdir="outputs", top_k=20, min_cluster_size=2):
    """Main function for entity consistency evaluation"""
    print(f"[Info] Reading nodes from: {nodes_path}")

    # Read nodes
    nodes = pd.read_csv(nodes_path)
    required = {"id", "label", "type"}
    if not required.issubset(nodes.columns):
        raise ValueError(f"nodes.csv must contain columns: {required}. Got: {nodes.columns.tolist()}")

    # Embedding + Clustering
    emb = compute_embeddings(nodes["label"], model_name)
    cluster_ids = cluster_with_version_compat(emb, threshold)
    nodes = nodes.copy()
    nodes["cluster_id"] = cluster_ids

    # Calculate metrics
    metrics = compute_redundancy_consistency(nodes)
    print("\n==== Entity Consistency Evaluation Results ====")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Export tables + visualization
    df_full, df_amb = export_cluster_tables(nodes, outdir, top_k=top_k, min_cluster_size=min_cluster_size)
    visualize(outdir, df_full, df_amb)

    # Save metrics json
    os.makedirs(outdir, exist_ok=True)
    metrics_path = os.path.join(outdir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[Save] {metrics_path}")

    print("\n[Done] Entity consistency results saved under:", os.path.abspath(outdir))
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Compute Unique Entity Rate / Type Consistency with clustering + visuals")
    parser.add_argument("--nodes", type=str, required=True, help="Path to nodes.csv (must include columns: id,label,type)")
    parser.add_argument("--threshold", type=float, default=0.35, help="Agglomerative distance_threshold (smaller = stricter)")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--outdir", type=str, default="outputs", help="Directory to save CSVs and plots")
    parser.add_argument("--top_k", type=int, default=20, help="How many most ambiguous clusters to export/plot")
    parser.add_argument("--min_cluster_size", type=int, default=2, help="Only clusters with size >= this for ambiguity list")
    args = parser.parse_args()

    evaluate_entity_consistency(args.nodes, args.threshold, args.model, args.outdir, args.top_k, args.min_cluster_size)

if __name__ == "__main__":
    main()
