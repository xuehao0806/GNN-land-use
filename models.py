import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import torch

def get_model(model_name, num_features, hidden_size, num_classes, device):
    if model_name == 'GraphSAGE':
        model = GraphSAGEModel(num_features, hidden_size, num_classes).to(device)
    elif model_name == 'GCN':
        model = GCNModel(num_features, hidden_size, num_classes).to(device)
    elif model_name == 'GAT':
        model = GATModel(num_features, hidden_size, num_classes).to(device)
    else:
        raise ValueError(f"Model {model_name} is not supported")
    return model

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.3)
        x = self.conv2(x, edge_index).relu()
        x = self.lin(x)
        return x


class GCNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.3)
        x = self.conv2(x, edge_index).relu()
        x = self.lin(x)
        return x


class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=2)
        self.conv2 = GATConv(hidden_channels * 2, hidden_channels, heads=1)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.3)
        x = self.conv2(x, edge_index).relu()
        x = self.lin(x)
        return x