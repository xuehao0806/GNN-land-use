import pandas as pd
import numpy as np
import torch
from torch_geometric.loader import RandomNodeSampler, NeighborLoader
from torch.optim import AdamW
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from utils import remove_reserved_nodes_and_edges
from models import get_model
## DATA LOADING
data = torch.load('data/processed/data.pt')

## MODELLING
# 1. parameters
num_features = data.x.shape[1]
hidden_size = 128
num_classes = data.y.shape[1]
num_heads = 2
depth = 1  # 采样深度
num_neighbors = 5  # 每个节点采样的邻居数
model_name = 'GCN'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sampler_name = 'RandomNode'

data_for_train_val = remove_reserved_nodes_and_edges(data, data.test_mask)
kwargs = {'batch_size': 64, 'num_workers': 2, 'persistent_workers': True, 'directed': False}
# Initialize NeighborLoader for training, validation, and testing
# The loader fetches neighbors up to 1 layer deep with a maximum of 10 neighbors for each node
train_loader = NeighborLoader(data_for_train_val, num_neighbors=[10] * 1, input_nodes=data.train_mask, **kwargs)
val_loader = NeighborLoader(data, num_neighbors=[10] * 1, input_nodes=data.val_mask, **kwargs)
test_loader = NeighborLoader(data, num_neighbors=[10] * 1, input_nodes=data.test_mask, **kwargs)


## MODELLING
num_features = data.x.shape[1]
num_classes = data.y.shape[1]
model = get_model(model_name, num_features, hidden_size, num_classes, device)
optimizer = AdamW(model.parameters(), lr=0.001)

# 3. models training
def train():
    model.train()
    total_loss = total_examples = 0
    loss_fn = torch.nn.MSELoss()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        # Assuming your model can handle node-level batches directly:
        out = model(batch.x, batch.edge_index)
        loss = loss_fn(out.squeeze(), batch.y.float())
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * batch.num_nodes  # Assuming loss is computed per node
        total_examples += batch.num_nodes
    return total_loss / total_examples

@torch.no_grad()
def test(loader):
    model.eval()
    total_loss = total_examples = 0
    loss_fn = torch.nn.MSELoss()
    for batch in loader:
        batch = batch.to(device)
        # Again, assuming the model can handle node-level batches:
        out = model(batch.x, batch.edge_index)
        loss = loss_fn(out.squeeze(), batch.y.float())
        total_loss += float(loss) * batch.num_nodes  # Assuming loss is computed per node
        total_examples += batch.num_nodes
    return total_loss / total_examples

for epoch in range(1, 21):
    loss = train()
    val_acc = test(val_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, ',
          f'Val: {val_acc:.4f} Test: {test_acc:.4f}')

## EVALUATION & SAVING
# 1. evaluation
def evaluation(loader):
    model.eval()
    predictions = {i: [] for i in range(6)}  # 6 indicators
    actuals = {i: [] for i in range(6)}

    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        for i in range(6):
            predictions[i].extend(out[:, i].cpu().detach().numpy())
            actuals[i].extend(data.y[:, i].cpu().numpy())

    indicators = ['office', 'sustenance', 'transport', 'retail', 'leisure', 'residence']
    metrics = ['MSE', 'RMSE', 'MAE', 'R2']

    results_df = pd.DataFrame(columns=metrics)

    for i, indicator in enumerate(indicators):
        mse = mean_squared_error(actuals[i], predictions[i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals[i], predictions[i])
        r2 = r2_score(actuals[i], predictions[i])

        results_df.loc[indicator] = [round(mse, 3), round(rmse, 3), round(mae, 3), round(r2, 3)]

    return results_df
results = evaluation(test_loader)
print(results)