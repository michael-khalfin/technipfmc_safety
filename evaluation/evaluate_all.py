import os
import argparse
from entity_consistency_eval import evaluate_entity_consistency
from link_prediction_holdout import evaluate_link_prediction
from semantic_similar_distance import evaluate_semantic_similarity

def main():
    parser = argparse.ArgumentParser(description="Run all evaluation methods")
    parser.add_argument("kg_dir", type=str, help="Directory containing the knowledge graph files (nodes.csv and edges.csv)")
    parser.add_argument("--outdir", type=str, default=None, help="Directory to save outputs")
    args = parser.parse_args()

    # Build file paths
    nodes_path = os.path.join(args.kg_dir, "nodes.csv")
    edges_path = os.path.join(args.kg_dir, "edges.csv")
    if args.outdir is None:
        args.outdir = os.path.join(args.kg_dir, "output")

    
    # Ensure input files exist
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

    # 2. Link Prediction Evaluation
    print("\n2. Running Link Prediction Evaluation...")
    link_metrics = evaluate_link_prediction(
        nodes_path=nodes_path,
        edges_path=edges_path,
        outdir=os.path.join(args.outdir, "link_prediction")
    )

    # 3. Semantic Similarity Evaluation
    print("\n3. Running Semantic Similarity Evaluation...")
    semantic_metrics = evaluate_semantic_similarity(
        nodes_path=nodes_path,
        edges_path=edges_path,
        outdir=os.path.join(args.outdir, "semantic_similarity")
    )

    # Collect all results and save to a unified file
    all_results = {
        "entity_consistency": entity_metrics,
        "link_prediction": link_metrics,
        "semantic_similarity": semantic_metrics
    }

    # Save results in JSON format
    import json
    os.makedirs(args.outdir, exist_ok=True)
    results_json = os.path.join(args.outdir, "all_evaluation_results.json")
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Save results in human-readable text format
    results_txt = os.path.join(args.outdir, "all_evaluation_results.txt")
    with open(results_txt, "w", encoding="utf-8") as f:
        # Entity Consistency Results
        f.write("="*50 + "\n")
        f.write("Entity Consistency Evaluation Results\n")
        f.write("="*50 + "\n")
        for k, v in entity_metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")

        # Link Prediction Results
        f.write("="*50 + "\n")
        f.write("Link Prediction (Holdout) Results\n")
        f.write("="*50 + "\n")
        methods = ["Adamic-Adar", "Jaccard", "ResourceAlloc"]
        for method, metrics in link_metrics.items():
            f.write(f"{method:>20s} | ")
            f.write(f"PR-AUC: {metrics['PR-AUC']:.4f} | ")
            f.write(f"ROC-AUC: {metrics['ROC-AUC']:.4f} | ")
            f.write(f"P@K: {metrics['P@K']:.4f} | ")
            f.write(f"R@K: {metrics['R@K']:.4f}\n")
        f.write("\n")

        # Semantic Similarity Results
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
    print(f"- Link Prediction: {os.path.join(args.outdir, 'link_prediction')}")
    print(f"- Semantic Similarity: {os.path.join(args.outdir, 'semantic_similarity')}")

if __name__ == "__main__":
    main()