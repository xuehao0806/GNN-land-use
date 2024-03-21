import torch
from torch.optim import AdamW
from torch.nn import Linear, ReLU
from torch_geometric.nn import SAGEConv, to_hetero, Sequential
from torch_geometric.loader import HGTLoader
from utils import get_loader, train_hetero, test, evaluation
import argparse

## PARAMETERS
# Create an argument parser object
parser = argparse.ArgumentParser(description='Train a GNN for land use identification.')
# Add arguments to the parser.
parser.add_argument('--loader_name', type=str, default='HGT', choices=['Neighbor', 'RandomNode','HGT'], help='Data loader to use (Neighbor or RandomNode).')
parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of the model.')
parser.add_argument('--model_name', type=str, default='GCN', choices=['GraphSAGE', 'GCN', 'GAT','NN'], help='Model to use for training.')
parser.add_argument('--learning_rate', type=float, default=0.002, help='Learning rate for the optimizer.')
parser.add_argument('--epoch_num', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--data_path', type=str, default='data/processed/data_hetero.pt', help='Path to the data file.')
parser.add_argument('--model_save_path', type=str, default='models/', help='Path where the trained model should be saved.')

# Parse the command-line arguments
args = parser.parse_args()

data_path = args.data_path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = torch.load(data_path)
print(f"Data objects: {data}")

# 定义每种节点和边类型的采样数
num_samples = {
    'bike': [10],  # 对于每个 bike 节点，第一跳采样10个邻居
    'bus': [10],   # 对于每个 bus 节点，第一跳采样15个邻居
    'tube': [10],   # 对于每个 tube 节点，第一跳采样5个邻居
}

kwargs = {'num_workers': 6, 'persistent_workers': True}
# 初始化HGTLoader
train_loader = HGTLoader(
    data,
    num_samples=num_samples,
    batch_size=32,  # 您可能需要调整这个值
    input_nodes=('bus', data['bus'].train_mask),**kwargs
)
val_loader = HGTLoader(
    data,
    num_samples=num_samples,
    batch_size=32,  # 您可能需要调整这个值
    input_nodes=('bus', data['bus'].val_mask),**kwargs
)
test_loader = HGTLoader(
    data,
    num_samples=num_samples,
    batch_size=32,  # 您可能需要调整这个值
    input_nodes=('bus', data['bus'].test_mask),**kwargs
)

class GraphSAGENet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGENet, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.nn.F.relu(x)
        x = self.conv2(x, edge_index)
        return x


def train(model, device, loader, optimizer):
    model.train()
    total_loss = 0

    for batch in loader:
        batch.to(device)
        optimizer.zero_grad()

        # 仅处理'bus'类型的节点
        if 'bus' in batch and batch['bus'].x is not None:
            bus_data = batch['bus']
            out = model(bus_data.x, bus_data.edge_index)
            loss = torch.nn.F.mse_loss(out, bus_data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * bus_data.num_nodes

    return total_loss / loader.num_samples

num_features = data['bus'].x.shape[1]
num_classes = data['bus'].y.shape[1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGENet(in_channels=num_features, hidden_channels=64, out_channels=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# 训练模型
num_epochs = 20
for epoch in range(num_epochs):
    loss = train(model, device, train_loader, optimizer)
    print(f'Epoch {epoch+1}, Loss: {loss}')