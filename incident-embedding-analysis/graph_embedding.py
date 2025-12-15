import pandas as pd
import numpy as np
import torch
import pickle
from torch_geometric.nn import Node2Vec

"""
Graph-based incident embedding using Node2Vec.

Pipeline:
1. Load incident-level graph triples
2. Build an undirected graph
3. Train Node2Vec on the graph
4. Map node embeddings to incident embeddings
5. Save incident-level embeddings

Assumption:
Each incident is represented by its source node in the graph.
"""


# =========================
# Configuration
# =========================
TRIPLE_PATH = "incident_triples.csv"      # CSV containing incident graph triples
OUT_PATH = "event_graph_embedding.pkl"    # Final output: incident_id -> embedding

EMBEDDING_DIM = 128
WALK_LENGTH = 40
CONTEXT_SIZE = 20
WALKS_PER_NODE = 10
NUM_NEGATIVE_SAMPLES = 1
P = 1.0
Q = 1.0
EPOCHS = 5
BATCH_SIZE = 128
LR = 0.01


# =========================
# 1. Load triples and build graph
# =========================
df = pd.read_csv(TRIPLE_PATH)
# Expected columns: incident_id, source, target

# Build node -> index mapping
nodes = pd.concat([df["source"], df["target"]]).unique()
node2idx = {node: idx for idx, node in enumerate(nodes)}

# Convert edges to integer indices
src = df["source"].map(node2idx).to_numpy()
dst = df["target"].map(node2idx).to_numpy()

# Undirected graph: add reverse edges
edge_index = np.vstack([
    np.concatenate([src, dst]),
    np.concatenate([dst, src])
])
edge_index = torch.from_numpy(edge_index).long()


# =========================
# 2. Train Node2Vec
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Node2Vec(
    edge_index=edge_index,
    embedding_dim=EMBEDDING_DIM,
    walk_length=WALK_LENGTH,
    context_size=CONTEXT_SIZE,
    walks_per_node=WALKS_PER_NODE,
    num_negative_samples=NUM_NEGATIVE_SAMPLES,
    p=P,
    q=Q,
    sparse=True
).to(device)

loader = model.loader(batch_size=BATCH_SIZE, shuffle=True)
optimizer = torch.optim.SparseAdam(model.parameters(), lr=LR)


def train_epoch():
    model.train()
    total_loss = 0.0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


for epoch in range(1, EPOCHS + 1):
    loss = train_epoch()
    print(f"Epoch {epoch}/{EPOCHS}, Loss: {loss:.4f}")


# =========================
# 3. Extract node embeddings
# =========================
node_embeddings = model.embedding.weight.data.cpu().numpy()
# shape: (num_nodes, EMBEDDING_DIM)


# =========================
# 4. Map node embeddings to incident embeddings
# =========================
event_embeddings = {}

for _, row in df.iterrows():
    incident_id = row["incident_id"]
    source_node = row["source"]

    if source_node not in node2idx:
        continue

    node_idx = node2idx[source_node]
    event_embeddings[incident_id] = node_embeddings[node_idx]


# =========================
# 5. Save incident-level embeddings
# =========================
with open(OUT_PATH, "wb") as f:
    pickle.dump(event_embeddings, f)

print(f"Saved {len(event_embeddings)} incident embeddings to {OUT_PATH}")
