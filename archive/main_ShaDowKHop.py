import pandas as pd
import numpy as np
import torch
from torch_geometric.loader import RandomNodeSampler, ShaDowKHopSampler
from torch.optim import AdamW
from models import GraphSAGEModel_s, GCNModel_s, GATModel_s, GraphSAGEModel_r, GCNModel_r, GATModel_r
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from utils import initialize_loaders, train, test, remove_reserved_nodes_and_edges

## DATA LOADING
data = torch.load('data/processed/data.pt')
kwargs = {'batch_size': 64, 'num_workers': 2, 'persistent_workers': True}
# build dataloader for training and
# removed edges and nodes data including the test nodes sets 预处理数据，移除测试集节点及其边
data_for_train_val = remove_reserved_nodes_and_edges(data, data.test_mask)
# building of training and testing datasets
train_loader = ShaDowKHopSampler(data_for_train_val, depth=2, num_neighbors=100,
                                 node_idx=data.train_mask, **kwargs)
val_loader = ShaDowKHopSampler(data, depth=2, num_neighbors=100,
                               node_idx=data.val_mask, **kwargs)
# building of testing datasets
test_loader = ShaDowKHopSampler(data, depth=2, num_neighbors=100,
                                node_idx=data.test_mask, **kwargs)


## MODELLING
# 1. parameters
num_features = data.x.shape[1]
hidden_size = 128
num_classes = data.y.shape[1]
num_heads = 2
depth = 1
num_neighbors = 5
model_name = 'GAT'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sampler_name = 'ShaDowKHop'


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

# Instantiate the model if the model class was found
if model_class:
    model = model_class(num_features, hidden_size, num_classes).to(device)
else:
    raise ValueError(f"Model {model_name} with sampler {sampler_name} is not supported.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = AdamW(model.parameters(), lr=0.0005)

# 3. models training
def train():
    model.train()
    total_loss = total_examples = 0
    loss_fn = torch.nn.MSELoss()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch, data.root_n_id)
        loss = loss_fn(out.squeeze(), data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_nodes
        total_examples += data.num_nodes
    return total_loss / total_examples


@torch.no_grad()
def test(loader):
    model.eval()
    total_loss = total_examples = 0
    loss_fn = torch.nn.MSELoss()
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch, data.root_n_id)
        loss = loss_fn(out.squeeze(), data.y.float())
        total_loss += float(loss) * data.num_nodes
        total_examples += data.num_nodes
    return total_loss / total_examples


for epoch in range(1, 21):
    loss = train()
    val_acc = test(val_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, ',
          f'Val: {val_acc:.4f} Test: {test_acc:.4f}')

## EVALUATION & SAVING
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # 避免除以零
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

# 1. evaluation
def evaluation(loader):
    model.eval()
    predictions = {i: [] for i in range(6)}  # Assume 6 indicators
    actuals = {i: [] for i in range(6)}

    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch, data.root_n_id)
        for i in range(6):
            predictions[i].extend(out[:, i].cpu().detach().numpy())
            actuals[i].extend(data.y[:, i].cpu().numpy())

    indicators = ['office', 'sustenance', 'transport', 'retail', 'leisure', 'residence']
    metrics = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2']

    results_df = pd.DataFrame(columns=metrics)

    for i, indicator in enumerate(indicators):
        mse = mean_squared_error(actuals[i], predictions[i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals[i], predictions[i])
        mape = mean_absolute_percentage_error(actuals[i], predictions[i])
        r2 = r2_score(actuals[i], predictions[i])

        results_df.loc[indicator] = [round(mse, 3), round(rmse, 3), round(mae, 3), round(mape, 3), round(r2, 3)]

    return results_df
results = evaluation(test_loader)
print(results)