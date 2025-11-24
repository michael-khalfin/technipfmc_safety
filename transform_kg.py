import pandas as pd
import os
from typing import Dict, List
import numpy as np
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime, timedelta
import json

# Entity types for normalized safety incident extraction
VALID_TYPES = {
    'INCIDENT_TYPE',    # fall, collision, spill, fire, struck-by, caught-in, near-miss
    'INJURY_TYPE',      # cut, bruise, burn, sprain, fracture (actual injuries only)
    'BODY_PART',        # hand, finger, back, leg, head, eye
    'EQUIPMENT',        # forklift, crane, ladder, drill, saw, truck
    'LOCATION',         # Specific facility/building names
    'ORGANIZATION',     # Company/contractor names
    'DATE',             # Incident dates
    'OTHER'             # Unclassifiable entities
}

# Checkpoint configuration
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def get_entity_type(entity: str) -> str:
    """Use local Ollama + mistral for entity type classification (single label)"""
    try:
        url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        payload = {
            "model": "mistral:latest",
            "prompt": (
                "You are an expert safety incident entity classifier.\n"
                "Classify each entity into EXACTLY ONE of these types:\n"
                "INCIDENT_TYPE, INJURY_TYPE, BODY_PART, EQUIPMENT, LOCATION, ORGANIZATION, DATE, OTHER.\n"
                "\n"
                "Rules:\n"
                "- Output ONLY the type name (all caps).\n"
                "- Do NOT add explanations or punctuation.\n"
                "- INCIDENT_TYPE: fall, collision, spill, fire, struck-by, caught-in, near-miss\n"
                "- INJURY_TYPE: cut, bruise, burn, sprain, fracture (actual injuries only)\n"
                "- BODY_PART: hand, finger, back, leg, head, eye\n"
                "- EQUIPMENT: forklift, crane, ladder, drill, saw, truck\n"
                "- LOCATION: Specific facility/building names (not generic 'workplace')\n"
                "- ORGANIZATION: Company/contractor names\n"
                "- DATE: Incident dates or times\n"
                "- OTHER: Generic terms or unclassifiable entities\n"
                "\n"
                "Examples:\n"
                "Entity: fall\nType: INCIDENT_TYPE\n\n"
                "Entity: forklift\nType: EQUIPMENT\n\n"
                "Entity: warehouse A\nType: LOCATION\n\n"
                "Entity: left hand\nType: BODY_PART\n\n"
                "Entity: deep cut\nType: INJURY_TYPE\n\n"
                "Entity: ABC Construction\nType: ORGANIZATION\n\n"
                "Entity: 2024-01-15\nType: DATE\n\n"
                "Entity: workplace\nType: OTHER\n\n"
                f"Entity: {entity}\nType:"
            ),
            "stream": False,
            "options": {
                "temperature": 0,
                "top_p": 1,
                "repeat_penalty": 1.1,
                "num_ctx": 2048
            }
        }
        resp = requests.post(f"{url}/api/generate", json=payload, timeout=30)
        if resp.status_code == 200:
            result = resp.json().get("response", "").strip()
            return result if result in VALID_TYPES else "OTHER"
        else:
            # Print error message, 404 usually means "model not found"
            print(f"Ollama error {resp.status_code}: {resp.text}")
            if resp.status_code == 404:
                print("Tip: Model not found locally, please run `ollama pull mistral` or check the model name.")
            return "OTHER"
    except Exception as e:
        print(f"Error getting type for entity {entity}: {e}")
        return "OTHER"


def load_checkpoint(checkpoint_name: str) -> Dict[str, str]:
    """Load previously saved entity type mappings from checkpoint"""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_name}.json")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_checkpoint(entity_types: Dict[str, str], checkpoint_name: str):
    """Save entity type mappings to checkpoint file"""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_name}.json")
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(entity_types, f, ensure_ascii=False, indent=2)
    print(f"Checkpoint saved: {checkpoint_path}")

def batch_get_entity_types(entities: List[str], batch_size: int = 4, use_full_dataset: bool = False, checkpoint_name: str = "entity_types") -> Dict[str, str]:
    """Batch process entity type classification using thread pool for efficiency.
    
    Args:
        entities: List of entities to classify
        batch_size: Number of concurrent threads
        use_full_dataset: If True, process all entities. If False, process 1% sample (default)
        checkpoint_name: Name of checkpoint file for resume capability
    """
    entity_types = {}
    valid_entities = [e for e in entities if pd.notna(e)]
    total_entities = len(valid_entities)
    
    # Load checkpoint if exists
    entity_types = load_checkpoint(checkpoint_name)
    already_classified = len(entity_types)
    
    # Select entities for processing
    if use_full_dataset:
        selected_entities = [e for e in valid_entities if e not in entity_types]
        sample_size = len(selected_entities)
    else:
        # Randomly select 1% of entities for processing
        sample_size = max(1, int(total_entities * 0.01))  # Process at least 1
        np.random.seed(42)  # Set random seed to maintain consistent results
        selected_entities = np.random.choice(valid_entities, size=sample_size, replace=False)
        # Filter out already classified
        selected_entities = [e for e in selected_entities if e not in entity_types]
    
    print(f"\nStarting entity type classification:")
    print(f"Total unique entities: {total_entities}")
    print(f"Already classified (from checkpoint): {already_classified}")
    print(f"Remaining to process: {sample_size}")
    print(f"Processing {len(selected_entities)} entities ({(len(selected_entities)/total_entities*100):.1f}% of total)")
    
    if len(selected_entities) == 0:
        print("All entities already classified. Skipping LLM processing.")
        return entity_types
    
    # Estimate total processing time
    estimated_time_per_entity = 0.5  # Assume each entity LLM processing takes ~0.5 seconds
    estimated_total_seconds = len(selected_entities) * estimated_time_per_entity
    estimated_completion_time = datetime.now() + timedelta(seconds=estimated_total_seconds)
    print(f"Estimated completion time: {estimated_completion_time.strftime('%H:%M:%S')}")
    print(f"Estimated duration: {timedelta(seconds=estimated_total_seconds)}\n")
    
    # Only process LLM for selected samples
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        future_to_entity = {
            executor.submit(get_entity_type, entity): entity 
            for entity in selected_entities
        }
        
        # tqdm
        with tqdm(total=len(selected_entities), desc="Classifying sample entities", 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            for future in as_completed(future_to_entity):
                entity = future_to_entity[future]
                try:
                    entity_type = future.result()
                    entity_types[entity] = entity_type
                    pbar.update(1)
                    pbar.set_postfix({"Last": entity[:20], "Type": entity_type})
                    # Save checkpoint every 50 entities
                    if pbar.n % 50 == 0:
                        save_checkpoint(entity_types, checkpoint_name)
                except Exception as e:
                    print(f"\nError processing entity {entity}: {e}")
                    pbar.update(1)
                # Add small delay to avoid excessive requests
                time.sleep(0.1)
    
    # Save final checkpoint
    save_checkpoint(entity_types, checkpoint_name)
    
    print("\nEntity classification completed!")
    

    type_counts = pd.Series(entity_types.values()).value_counts()
    print("\nType distribution:")
    for type_name, count in type_counts.items():
        print(f"{type_name}: {count} ({count/len(entity_types)*100:.1f}%)")
    
    print(f"\nNote: {total_entities - len(entity_types)} entities were not classified (only sampled/checkpoint entities are included)")
    
    return entity_types

def process_nodes(input_path: str, output_path: str, use_full_dataset: bool = False, checkpoint_name: str = "entity_types"):
    """Process nodes.csv file and add type field
    
    Args:
        input_path: Path to input nodes.csv
        output_path: Path to output nodes.csv
        use_full_dataset: If True, classify all entities. If False, use 1% sample (default)
        checkpoint_name: Name of checkpoint file for resume capability
    """
    print("Processing nodes...")
    nodes_df = pd.read_csv(input_path)
    
    # Get all entities
    unique_entities = nodes_df['entity'].dropna().unique()
    print(f"Processing {len(unique_entities)} unique entities...")
    print(f"Full dataset mode: {use_full_dataset}")
    
    # Batch process entity types
    entity_type_map = batch_get_entity_types(unique_entities, use_full_dataset=use_full_dataset, checkpoint_name=checkpoint_name)
    

    # Create mapping from entity to type
    # Use 'Other' as fallback for entities not classified by LLM
    entity_types = {}
    for _, row in nodes_df.iterrows():
        if pd.notna(row['entity']):
            entity_types[row['id']] = entity_type_map.get(row['entity'], 'Other')
    
    # Create new DataFrame
    new_nodes_df = pd.DataFrame({
        'id': nodes_df['id'],
        'label': nodes_df['entity'],
        'type': nodes_df['id'].map(entity_types)
    })
    
    print(f"Entity type distribution in output:")
    type_dist = new_nodes_df['type'].value_counts()
    for entity_type, count in type_dist.items():
        print(f"  {entity_type}: {count} ({count/len(new_nodes_df)*100:.1f}%)")
    
    new_nodes_df.to_csv(output_path, index=False)
    return entity_types

def process_edges(edges_path: str, nodes_df: pd.DataFrame, kg_edges_path: str, output_path: str):
    """Process edges.csv file and add labels and row_id"""
    print("Processing edges...")
    edges_df = pd.read_csv(edges_path)
    kg_edges_df = pd.read_csv(kg_edges_path)
    
    # Create mapping from node id to label
    node_labels = dict(zip(nodes_df['id'], nodes_df['entity']))
    
    # Create new DataFrame
    new_edges_df = pd.DataFrame({
        'src': edges_df['source'],
        'rel': edges_df['relation'],
        'dst': edges_df['target'],
        'src_label': edges_df['source'].map(node_labels),
        'dst_label': edges_df['target'].map(node_labels),
        'row_id': range(1, len(edges_df) + 1)  # Use auto-increment ID
    })
    
    new_edges_df.to_csv(output_path, index=False)

def main(use_full_dataset: bool = False):
    """Main function to transform KG
    
    Args:
        use_full_dataset: If True, process all entities. If False, use 1% sample (default)
                         Set to True for full-scale runs, False for testing
    """
    # Create output directory
    output_dir = "trans"
    os.makedirs(output_dir, exist_ok=True)
    
    # Input file paths
    kg2_dir = "KG2"
    input_nodes = os.path.join(kg2_dir, "nodes.csv")
    input_edges = os.path.join(kg2_dir, "edges.csv")
    input_kg_edges = os.path.join(kg2_dir, "knowledge_graph_edges.csv")
    
    # Output file paths
    output_nodes = os.path.join(output_dir, "nodes.csv")
    output_edges = os.path.join(output_dir, "edges.csv")
    
    # Checkpoint name
    checkpoint_name = "entity_types_full" if use_full_dataset else "entity_types_sample"
    
    # Process nodes
    nodes_df = pd.read_csv(input_nodes)
    entity_types = process_nodes(input_nodes, output_nodes, use_full_dataset=use_full_dataset, checkpoint_name=checkpoint_name)
    
    # Process edges
    process_edges(input_edges, nodes_df, input_kg_edges, output_edges)
    
    print("Transformation completed. Files saved in 'trans' directory.")

if __name__ == "__main__":
    import sys
    # Use full dataset if --full flag is provided
    use_full = "--full" in sys.argv
    if use_full:
        print("Running in FULL DATASET mode - all entities will be classified")
        print("Checkpoints will be saved to: checkpoints/entity_types_full.json")
    else:
        print("Running in SAMPLE mode - 1% of entities will be classified")
        print("Tip: Run with --full flag to process all entities: python transform_kg.py --full")
        print("Checkpoints will be saved to: checkpoints/entity_types_sample.json")
    
    # Check for existing checkpoints
    checkpoint_name = "entity_types_full" if use_full else "entity_types_sample"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_name}.json")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        print(f"\nFound existing checkpoint with {len(existing)} classified entities.")
        print(f"Checkpoint file: {checkpoint_path}")
        print("Resuming from last checkpoint...\n")
    
    main(use_full_dataset=use_full)