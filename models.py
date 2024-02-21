import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import torch

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_size)
        self.conv2 = SAGEConv(hidden_size, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class GCNModel(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_feats, hidden_size)
        self.conv2 = GCNConv(hidden_size, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class GATModel(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, num_heads):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_feats, hidden_size, heads=num_heads)
        self.conv2 = GATConv(hidden_size * num_heads, num_classes, heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x