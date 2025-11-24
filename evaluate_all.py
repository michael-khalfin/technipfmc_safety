import os
import argparse
from entity_consistency_eval import evaluate_entity_consistency
from semantic_similar_distance import evaluate_semantic_similarity
import pandas as pd
import networkx as nx
from networkx.algorithms import community as nx_comm
from collections import Counter, defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def evaluate_topology(nodes_path, edges_path, outdir):
    # Load data
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    # Build graph
    Gd = nx.DiGraph()
    nid = nodes_df.columns[0]
    for _, r in nodes_df.iterrows():
        Gd.add_node(r[nid], **r.drop(labels=[nid]).to_dict())
    a_col, b_col = edges_df.columns[0], edges_df.columns[1]
    for _, r in edges_df.iterrows():
        Gd.add_edge(r[a_col], r[b_col])
    G = Gd.to_undirected(reciprocal=False)
    # Metrics
    def giant_component_stats(G):
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
    def path_stats(G):
        if G.number_of_nodes() == 0:
            return {"avg_shortest_path": None, "diameter": None}
        comps = list(nx.connected_components(G))
        largest = max(comps, key=len)
        if len(largest) <= 1:
            return {"avg_shortest_path": None, "diameter": None}
        Gsub = G.subgraph(largest)
        try:
            avg = nx.average_shortest_path_length(Gsub)
        except Exception:
            avg = None
        try:
            diam = nx.diameter(Gsub)
        except Exception:
            diam = None
        return {"avg_shortest_path": avg, "diameter": diam}
    def clustering_stats(G):
        if G.number_of_nodes() == 0:
            return {"avg_clustering": None, "global_transitivity": None}
        avg_c = nx.average_clustering(G)
        trans = nx.transitivity(G)
        return {"avg_clustering": avg_c, "global_transitivity": trans}
    def community_stats(G, top_n_communities=10):
        if G.number_of_nodes() == 0:
            return {"n_communities": 0, "modularity": None}
        try:
            communities = list(nx_comm.greedy_modularity_communities(G))
            modularity = nx_comm.modularity(G, communities)
        except Exception:
            communities = [set(G.nodes())]
            modularity = None
        top_comms = sorted([(i, len(c)) for i, c in enumerate(communities)], key=lambda x: x[1], reverse=True)[:top_n_communities]
        community_sizes = [len(c) for c in communities]
        return {
            "n_communities": len(communities),
            "modularity": modularity,
            "top_communities": [{"id": int(i), "size": int(s)} for i, s in top_comms],
            "community_sizes": community_sizes,
        }
    def degree_and_assortativity(Gd, G, nodes_df):
        degs = [d for _, d in G.degree()]
        if not degs:
            return {}
        mean_deg = float(sum(degs) / len(degs))
        median_deg = float(sorted(degs)[len(degs) // 2])
        max_deg = max(degs)
        frac_deg1 = sum(1 for d in degs if d == 1) / len(degs)
        # degree assortativity (numeric)
        try:
            from networkx.algorithms.assortativity import degree_pearsonr
            r, t, s = degree_pearsonr(G)
            denom = 1 - s
            if abs(denom) < 1e-8:
                denom += 1e-8
            assort = (t - s) / denom
        except Exception:
            try:
                assort = nx.degree_assortativity_coefficient(G)
            except Exception:
                assort = None
        return {
            "mean_degree": mean_deg,
            "median_degree": median_deg,
            "max_degree": int(max_deg),
            "frac_degree_1": float(frac_deg1),
            "degree_assortativity": assort,
        }
    metrics = {}
    metrics['connectivity'] = giant_component_stats(G)
    metrics['paths'] = path_stats(G)
    metrics['clustering'] = clustering_stats(G)
    metrics['community'] = community_stats(G)
    metrics['degree'] = degree_and_assortativity(Gd, G, nodes_df)
    # Save results
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "topology_metrics.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    # Save visualizations
    def save_plots(G, community_sizes, out_dir, prefix="topology"):
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
    
    try:
        community_sizes = metrics['community'].get('community_sizes', [])
        save_plots(G, community_sizes, outdir, prefix="topology")
    except Exception as e:
        print(f"Warning: failed to save plots: {e}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Run all evaluation methods")
    parser.add_argument("kg_dir", type=str, help="Directory containing the knowledge graph files (nodes.csv and edges.csv)")
    parser.add_argument("--outdir", type=str, default=None, help="Directory to save outputs")
    args = parser.parse_args()

    nodes_path = os.path.join(args.kg_dir, "nodes.csv")
    edges_path = os.path.join(args.kg_dir, "edges.csv")
    if args.outdir is None:
        args.outdir = os.path.join(args.kg_dir, "output")

    if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
        raise FileNotFoundError(f"Please ensure both nodes.csv and edges.csv exist in {args.kg_dir}")

    print("\n" + "="*50)
    print("Starting comprehensive evaluation...")
    print("="*50)

    # 1. Entity Consistency Evaluation
    print("\n1. Running Entity Consistency Evaluation...")
    entity_metrics = evaluate_entity_consistency(
        nodes_path=nodes_path,
        outdir=os.path.join(args.outdir, "entity_consistency")
    )

    # 2. Topology Robustness Evaluation
    print("\n2. Running Topology Robustness Evaluation...")
    topology_metrics = evaluate_topology(
        nodes_path,
        edges_path,
        os.path.join(args.outdir, "topology")
    )

    # 3. Semantic Similarity Evaluation
    print("\n3. Running Semantic Similarity Evaluation...")
    semantic_metrics = evaluate_semantic_similarity(
        nodes_path=nodes_path,
        edges_path=edges_path,
        outdir=os.path.join(args.outdir, "semantic_similarity")
    )

    all_results = {
        "entity_consistency": entity_metrics,
        "topology": topology_metrics,
        "semantic_similarity": semantic_metrics
    }

    import json
    os.makedirs(args.outdir, exist_ok=True)
    results_json = os.path.join(args.outdir, "all_evaluation_results.json")
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    results_txt = os.path.join(args.outdir, "all_evaluation_results.txt")
    with open(results_txt, "w", encoding="utf-8") as f:
        f.write("="*50 + "\n")
        f.write("Entity Consistency Evaluation Results\n")
        f.write("="*50 + "\n")
        for k, v in entity_metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")
        f.write("="*50 + "\n")
        f.write("Topology Robustness Evaluation Results\n")
        f.write("="*50 + "\n")
        for k, v in topology_metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")
        f.write("="*50 + "\n")
        f.write("Semantic Similarity Results\n")
        f.write("="*50 + "\n")
        f.write(f"Spearman r = {semantic_metrics['spearman_r']:.4f} ")
        f.write(f"(p={semantic_metrics['p_value']:.2e}), ")
        f.write(f"n={semantic_metrics['n_samples']}\n")

    print("\n" + "="*50)
    print("Evaluation Complete!")
    print("="*50)
    print("\nResults have been saved to:")
    print(f"- Detailed results (JSON): {results_json}")
    print(f"- Human readable results: {results_txt}")
    print("\nVisualization directories:")
    print(f"- Entity Consistency: {os.path.join(args.outdir, 'entity_consistency')}")
    print(f"- Topology: {os.path.join(args.outdir, 'topology')}")
    print(f"- Semantic Similarity: {os.path.join(args.outdir, 'semantic_similarity')}")

if __name__ == "__main__":
    main()