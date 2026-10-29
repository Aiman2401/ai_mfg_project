import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

# 1. Simple GraphSAGE Model Definition
class ImpactGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(ImpactGNN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)  # Returns impact probability [0, 1] for each node

# 2. Build PyTorch Geometric Data object from Neo4j / Local graph
def prepare_graph_data(nodes, edges, seed_node_id):
    """
    nodes: list of node dicts [{'id': ..., 'type': ...}]
    edges: list of tuples [(src_idx, dst_idx)]
    seed_node_id: ID of the changed component (e.g., 'FRAME-TOP-PLATE')
    """
    node_id_map = {n['id']: i for i, n in enumerate(nodes)}
    
    # Simple One-Hot / Distance Feature:
    # Feature 0: Is this the changed seed node? (1 if seed else 0)
    # Feature 1: Node type flag
    features = []
    for n in nodes:
        is_seed = 1.0 if n['id'] == seed_node_id else 0.0
        node_type_val = 1.0 if n.get('type') == 'Component' else 0.5
        features.append([is_seed, node_type_val])

    x = torch.tensor(features, dtype=torch.float)
    
    # Convert edge index
    edge_list = []
    for src, dst in edges:
        if src in node_id_map and dst in node_id_map:
            edge_list.append([node_id_map[src], node_id_map[dst]])
            edge_list.append([node_id_map[dst], node_id_map[src]]) # Undirected
            
    if not edge_list:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index), node_id_map

# 3. Prediction Function called by Agent
def predict_impact_scores(nodes, edges, seed_node_id):
    data, id_map = prepare_graph_data(nodes, edges, seed_node_id)
    
    # Instantiate lightweight model
    model = ImpactGNN(in_channels=2, hidden_channels=16)
    model.eval()
    
    with torch.no_grad():
        probs = model(data.x, data.edge_index).squeeze(-1).tolist()
    
    # Map node IDs back to predicted impact probability
    rev_map = {i: nid for nid, i in id_map.items()}
    predictions = {rev_map[i]: float(probs[i]) for i in range(len(probs))}
    
    return predictions
