import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from sklearn.preprocessing import StandardScaler
from models import GraphSAGEModel, GCNModel, GATModel
from utils import evaluate_metrics

## DATA PREPARATION
# 1. load nodes features data
# 1.1 basic features
bike_features = pd.read_csv('data/inputs/bike_15mins_filtered.csv')
bus_features = pd.read_csv('data/inputs/bus_15mins_filtered.csv')
tube_features = pd.read_csv('data/inputs/train_15mins_filtered.csv')
# 1.2 add nodes_type features
# 1.3 combine features (for homo-graph)
combined_features = pd.concat([bike_features, bus_features, tube_features], ignore_index=True)
# 1.4 normalisation
scaler = StandardScaler()
X = combined_features.drop('LocationID', axis=1).values
X_scaled = scaler.fit_transform(X)

# 2. load labels data
# 2.1 six types of land use labels
label_columns_list = ['office',	'sustenance', 'transport',	'retail', 'leisure', 'residence']
bike_labels = pd.read_csv('data/label/processed_merged_bikelocation_POI_filtered.csv')
bus_labels = pd.read_csv('data/label/processed_merged_buslocation_POI_filtered.csv')
tube_labels = pd.read_csv('data/label/processed_merged_trainlocation_POI_filtered.csv')
# 2.2 combine labels (for homo-graph)
combined_label = pd.concat([bike_labels, bus_labels, tube_labels], ignore_index=True)
# 2.3 normalisation
y = combined_label[label_columns_list].values

# 3. load edge features data
adj_matrix = pd.read_csv("data/edge/real_knn_edges.csv")
src_nodes = adj_matrix["Start"].values
dst_nodes = adj_matrix["End"].values
weights = adj_matrix["Weight_gk"].values

edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

# 4. build homo_DGL_graph
data = Data(x=torch.tensor(X_scaled, dtype=torch.float32), edge_index=edge_index)
data.edge_attr = torch.tensor(weights, dtype=torch.float32).view(-1, 1)
data.y = torch.tensor(y, dtype=torch.float32)

## MODELLING
# 1. parameters
num_features = X.shape[1]
hidden_size = 64
num_classes = y.shape[1]
num_heads = 2
model_name = 'GAT'
num_epochs = 1000

# 2. models building
if model_name == 'GraphSAGE':
    model = GraphSAGEModel(num_features, hidden_size, num_classes)
if model_name == 'GCN':
    model = GCNModel(num_features, hidden_size, num_classes)
if model_name == 'GAT':
    model = GATModel(num_features, hidden_size, num_classes, num_heads)

# 3. dataset separating and basic setting
optimizer = AdamW(model.parameters(), lr=0.002)
loss_fn = torch.nn.MSELoss()

train_mask, test_mask = train_test_split(np.arange(data.num_nodes), test_size=0.2, random_state=42)
train_mask = torch.tensor(train_mask, dtype=torch.long)
test_mask = torch.tensor(test_mask, dtype=torch.long)

# 4. training
for epoch in range(num_epochs):
    model.train()
    out = model(data)
    loss = loss_fn(out[train_mask], data.y[train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

## EVALUATION & SAVING
# 1. evaluation
results = {"Label": [], "MSE": [], "RMSE": [], "MAE": [], "R2 Score": []}

model.eval()
with torch.no_grad():
    out = model(data)
    test_loss = loss_fn(out[test_mask], data.y[test_mask])
    print(f"Test Loss: {test_loss.item()}")

    for i, column in enumerate(label_columns_list):
        mse, rmse, mae, r2 = evaluate_metrics(data.y[test_mask, i].numpy(),
                                              out[test_mask, i].numpy())

        results["Label"].append(column)
        results["MSE"].append(mse)
        results["RMSE"].append(rmse)
        results["MAE"].append(mae)
        results["R2 Score"].append(r2)

results_df = pd.DataFrame(results)
# 2. saving
# torch.save(model.state_dict(), f'models/{model_name}.pth')
# results_df.to_csv(f'evaluation/test_performance.csv', index=False)
print(results_df)