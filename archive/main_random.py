import pandas as pd
import numpy as np
import torch
from torch_geometric.loader import RandomNodeSampler, ShaDowKHopSampler
from torch.optim import AdamW
from models import GraphSAGEModel_s, GCNModel_s, GATModel_s, GraphSAGEModel_r, GCNModel_r, GATModel_r
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

## DATA LOADING
data = torch.load('data/processed/data.pt')

## MODELLING
# 1. parameters
num_features = data.x.shape[1]
hidden_size = 32
num_classes = data.y.shape[1]
num_heads = 2
depth = 1  # 采样深度
num_neighbors = 5  # 每个节点采样的邻居数
model_name = 'GCN'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sampler_name = 'RandomNode'

kwargs = {'num_workers': 4, 'persistent_workers': True}

train_num_parts = 60
val_num_parts = 20
test_num_parts = 20

train_loader = RandomNodeSampler(data, num_parts=train_num_parts, shuffle=True, **kwargs)
val_loader = RandomNodeSampler(data, num_parts=val_num_parts, shuffle=False, **kwargs)
test_loader = RandomNodeSampler(data, num_parts=test_num_parts, shuffle=False, **kwargs)

# 2. models building and statement
model_classes = {
    'RandomNode': {
        'GraphSAGE': GraphSAGEModel_r,
        'GCN': GCNModel_r,
        'GAT': GATModel_r,
    },
    'ShaDowKHop': {
        'GraphSAGE': GraphSAGEModel_s,
        'GCN': GCNModel_s,
        'GAT': GATModel_s,
    }
}
model_class = model_classes[sampler_name].get(model_name)

model_class = model_classes[sampler_name].get(model_name)
# Instantiate the model if the model class was found
if model_class:
    model = model_class(num_features, hidden_size, num_classes).to(device)
else:
    raise ValueError(f"Model {model_name} with sampler {sampler_name} is not supported.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = AdamW(model.parameters(), lr=0.002)

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