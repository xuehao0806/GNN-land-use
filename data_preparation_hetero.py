import torch
from torch_geometric.data import HeteroData
import argparse

parser = argparse.ArgumentParser(description='Train a hetero-GNN for land use identification.')
parser.add_argument('--data_path', type=str, default='data/processed/data_homo.pt', help='Path to the data file.')
args = parser.parse_args()
data_path = args.data_path
data = torch.load(data_path)

# Create a HeteroData object
hetero_data = HeteroData()

# Add node data
hetero_data['node'].x = data['x']
hetero_data['node'].y = data['y']
hetero_data['node'].train_mask = data['train_mask']
hetero_data['node'].val_mask = data['val_mask']
hetero_data['node'].test_mask = data['test_mask']

# Process edge data
edge_types = ['type0', 'type1', 'type2', 'type3', 'type4']
for i, edge_type in enumerate(edge_types):
    # Get indices of edges of the current type from edge_attr
    mask = data['edge_attr'][:, i + 1].bool()
    edge_indices = data['edge_index'][:, mask]

    # Add to the HeteroData object
    hetero_data['node', edge_type, 'node'].edge_index = edge_indices
    hetero_data['node', edge_type, 'node'].edge_attr = data['edge_attr'][mask, 0]  # Assume the first dimension is the feature of the edge

data_path = 'data/processed/data_hetero.pt'
torch.save(hetero_data, data_path)
print(hetero_data)

