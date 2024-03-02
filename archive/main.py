import pandas as pd
import numpy as np
import torch
from torch_geometric.loader import RandomNodeSampler, ShaDowKHopSampler
from torch.optim import AdamW
from models import GraphSAGEModel_s, GCNModel_s, GATModel_s, GraphSAGEModel_r, GCNModel_r, GATModel_r
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from utils import initialize_loaders, train, test

## DATA LOADING
data = torch.load('data/processed/data.pt')

## MODELLING
# 1. parameters
num_features = data.x.shape[1]
hidden_size = 32
num_classes = data.y.shape[1]
num_heads = 2
depth = 1  # sampling depth
num_neighbors = 5  # number of neighbors
model_name = 'GraphSAGE'
num_epochs = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sampler_name = 'RandomNode'

kwargs = {'batch_size': 64, 'num_workers': 4, 'persistent_workers': True}

# Initialize loaders using the function
loaders = initialize_loaders(sampler_name, data, kwargs)
train_loader, val_loader, test_loader = loaders['train_loader'], loaders['val_loader'], loaders['test_loader']

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
optimizer = AdamW(model.parameters(), lr=0.001)

for epoch in range(1, 31):
    loss = train(model, train_loader, optimizer, device, sampler_name)
    val_acc = test(model, val_loader, device, sampler_name)
    test_acc = test(model, test_loader, device, sampler_name)
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
        if sampler_name == 'ShadowKHop':
            out = model(data.x, data.edge_index, data.batch, data.root_n_id)
        elif sampler_name == 'RandomNode':
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