import torch
import torch.nn.functional as F
from torch_geometric.loader import HGTLoader
from torch_geometric.nn import HGTConv, Linear
from utils import evaluation
import pandas as pd

data = torch.load('data/processed/data_hetero.pt')
# print(f"Data objects: {data}")
kwargs = {'batch_size': 64, 'num_workers': 0, 'persistent_workers': False}
# Creating heterogeneous graph training, validation, and test loaders
train_loader = HGTLoader(data, num_samples={'node': [512] * 2}, shuffle=True,
                         input_nodes=('node', data['node'].train_mask), **kwargs)
val_loader = HGTLoader(data, num_samples={'node': [512] * 2},
                       input_nodes=('node', data['node'].val_mask), **kwargs)
test_loader = HGTLoader(data, num_samples={'node': [512] * 2},
                        input_nodes=('node', data['node'].test_mask), **kwargs)

class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict['node'])

model = HGT(hidden_channels=64, out_channels=6, num_heads=2, num_layers=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data, model = data.to(device), model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)

with torch.no_grad():  # Initialize lazy modules.
    out = model(data.x_dict, data.edge_index_dict)

def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x_dict, batch.edge_index_dict)
        loss = F.mse_loss(out[batch['node'].train_mask], batch['node'].y[batch['node'].train_mask].float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

@torch.no_grad()  # Disable gradient computation, reduce memory usage and computation needs
def test(loader):
    model.eval()  # Switch to evaluation mode
    mse = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x_dict, batch.edge_index_dict)
        mask = batch['node'].test_mask
        mse.append(F.mse_loss(pred[mask], batch['node'].y[mask].float()).item())
    return sum(mse) / len(loader)

for epoch in range(1, 201):
    loss = train()
    val_mse = test(test_loader)  # Should use a separate validation set loader
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val MSE: {val_mse:.4f}')

model_name = 'HGT'
loader_name = 'HGTLoader'
model_save_path = 'models/'
results = evaluation(model_name, model, test_loader, device)
torch.save(model.state_dict(), f'{model_save_path}{model_name}_{loader_name}.pth')
print(results)

# @torch.no_grad()  # Disable gradient computation, reduce memory usage
# def calculate_residuals(model, loader):
#     model.eval()  # Switch to evaluation mode
#     residuals = []

#     for batch in loader:
#         batch = batch.to(device)
#         pred = model(batch.x_dict, batch.edge_index_dict)  # Model prediction
#         true = batch['node'].y.float()  # True labels

#         # Calculate residuals: the difference between predictions and true values
#         residual = (pred - true).cpu().numpy()  # Move to CPU and convert to NumPy array
#         residuals.extend(residual)

#     # Combine all residuals into a DataFrame
#     residuals_df = pd.DataFrame(residuals, columns=['office', 'sustenance', 'transport', 'retail', 'leisure', 'residence'])
#     return residuals_df

# # Modify batch size, if memory is sufficient, can load all nodes at once
# kwargs = {'batch_size': 4236, 'num_workers': 0, 'persistent_workers': False}

# # Create a loader to load the entire dataset
# full_loader = HGTLoader(data, num_samples={'node': [4236]}, shuffle=False,
#                         input_nodes=('node', torch.ones(data['node'].num_nodes, dtype=bool)), **kwargs)

# residuals_df = calculate_residuals(model, full_loader)
# residuals_df.to_csv(f'./visualisation/residual/{model_name}_{loader_name}.csv')
