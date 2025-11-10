import pandas as pd
import os
from typing import Dict, List
import numpy as np
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime, timedelta

VALID_TYPES = {'Person', 'Equipment', 'Location', 'Event', 'Action', 'State', 'Other'}

def get_entity_type(entity: str) -> str:
    """Use local Ollama + mistral for entity type classification (single label)"""
    try:
        url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        payload = {
            "model": "mistral:latest",
            "prompt": (
                "You are an expert entity classifier.\n"
                "Classify the entity into EXACTLY ONE of these types:\n"
                "Person, Equipment, Location, Event, Action, State, Other.\n"
                "Rules:\n"
                "- Output ONLY the type name.\n"
                "- Do NOT add explanations or punctuation.\n\n"
                "Examples:\n"
                "Entity: John Smith\nType: Person\n\n"
                "Entity: hydraulic pump\nType: Equipment\n\n"
                "Entity: Houston\nType: Location\n\n"
                "Entity: evacuation\nType: Action\n\n"
                "Entity: shutdown\nType: Event\n\n"
                "Entity: overheating\nType: State\n\n"
                "Entity: Q1-2023 plan\nType: Other\n\n"
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
            return result if result in VALID_TYPES else "Other"
        else:
            # Print error message, 404 usually means "model not found"
            print(f"Ollama error {resp.status_code}: {resp.text}")
            if resp.status_code == 404:
                print("Tip: Model not found locally, please run `ollama pull mistral` or check the model name.")
            return "Other"
    except Exception as e:
        print(f"Error getting type for entity {entity}: {e}")
        return "Other"


def batch_get_entity_types(entities: List[str], batch_size: int = 4) -> Dict[str, str]:
    """Batch process entity type classification using thread pool for efficiency. Only process 1% of entities, mark others as Other"""
    entity_types = {}
    valid_entities = [e for e in entities if pd.notna(e)]
    total_entities = len(valid_entities)
    
    # Randomly select 1% of entities for processing
    sample_size = max(1, int(total_entities * 0.01))  # Process at least 1
    np.random.seed(42)  # Set random seed to maintain consistent results
    selected_entities = np.random.choice(valid_entities, size=sample_size, replace=False)
    
    print(f"\nStarting entity type classification:")
    print(f"Total unique entities: {total_entities}")
    print(f"Processing {sample_size} entities ({(sample_size/total_entities*100):.1f}% sample)")
    
    # Estimate total processing time
    estimated_time_per_entity = 0.5  # Assume each entity LLM processing takes ~0.5 seconds
    estimated_total_seconds = sample_size * estimated_time_per_entity
    estimated_completion_time = datetime.now() + timedelta(seconds=estimated_total_seconds)
    print(f"Estimated completion time: {estimated_completion_time.strftime('%H:%M:%S')}")
    print(f"Estimated duration: {timedelta(seconds=estimated_total_seconds)}\n")

    # Mark all entities as Other first
    entity_types = {entity: "Other" for entity in valid_entities}
    
    # Only process LLM for selected samples
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        future_to_entity = {
            executor.submit(get_entity_type, entity): entity 
            for entity in selected_entities
        }
        
        # 使用tqdm显示进度
        with tqdm(total=sample_size, desc="Classifying sample entities", 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            for future in as_completed(future_to_entity):
                entity = future_to_entity[future]
                try:
                    entity_type = future.result()
                    entity_types[entity] = entity_type
                    pbar.update(1)
                    pbar.set_postfix({"Last": entity[:20], "Type": entity_type})
                except Exception as e:
                    print(f"\nError processing entity {entity}: {e}")
                    pbar.update(1)
                # Add small delay to avoid excessive requests
                time.sleep(0.1)
    
    print("\nEntity classification completed!")
    
    # 显示统计信息
    type_counts = pd.Series(entity_types.values()).value_counts()
    print("\nType distribution:")
    for type_name, count in type_counts.items():
        print(f"{type_name}: {count} ({count/total_entities*100:.1f}%)")
    
    return entity_types

def process_nodes(input_path: str, output_path: str):
    """Process nodes.csv file and add type field"""
    print("Processing nodes...")
    nodes_df = pd.read_csv(input_path)
    
    # Get all entities
    unique_entities = nodes_df['entity'].dropna().unique()
    print(f"Processing {len(unique_entities)} unique entities...")
    
    # Batch process entity types
    entity_type_map = batch_get_entity_types(unique_entities)
    

    # Create mapping from entity to type
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

def main():
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
    
    # Process nodes
    nodes_df = pd.read_csv(input_nodes)
    entity_types = process_nodes(input_nodes, output_nodes)
    
    # Process edges
    process_edges(input_edges, nodes_df, input_kg_edges, output_edges)
    
    print("Transformation completed. Files saved in 'trans' directory.")

if __name__ == "__main__":
    main()