import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 1. load nodes features data
# 1.1 basic features
bike_features = pd.read_csv('data/inputs/bike_15mins_filtered.csv')
bus_features = pd.read_csv('data/inputs/bus_15mins_filtered.csv')
tube_features = pd.read_csv('data/inputs/train_15mins_filtered.csv')
# 1.2 add nodes_type features
bike_features['TransportType'] = 'bike'
bus_features['TransportType'] = 'bus'
tube_features['TransportType'] = 'tube'
# 1.3 combine features (for homo-graph)
combined_features = pd.concat([bike_features, bus_features, tube_features], ignore_index=True)
combined_features = pd.get_dummies(combined_features, columns=['TransportType'])
# 1.4 normalisation
scaler = StandardScaler()
X = combined_features.drop('LocationID', axis=1).values
X_scaled = scaler.fit_transform(X)

# 2. load labels data
# 2.1 six types of land use labels
label_columns_list = ['office',	'sustenance', 'transport',	'retail', 'leisure', 'residence']
bike_labels = pd.read_csv('data/label/0-1_processed_merged_bikelocation_POI_filtered_wr.csv')
bus_labels = pd.read_csv('data/label/0-1_processed_merged_buslocation_POI_filtered_wr.csv')
tube_labels = pd.read_csv('data/label/0-1_processed_merged_trainlocation_POI_filtered_wr.csv')
# 2.2 combine labels (for homo-graph)
combined_label = pd.concat([bike_labels, bus_labels, tube_labels], ignore_index=True)
# 2.3 normalisation
y = combined_label[label_columns_list].values
# 2.4 normalisationLoad edge data and create edge_index tensor
adj_matrix_highway = pd.read_csv("data/edge/real_knn_edges_highway.csv")
edge_index_np = np.vstack([adj_matrix_highway["Start"].values, adj_matrix_highway["End"].values])
edge_index = torch.tensor(edge_index_np, dtype=torch.long)

# 3. build graph data for model training
# 3.1 create graph data structure
encoder = OneHotEncoder()
waytype_encoded = encoder.fit_transform(adj_matrix_highway[['Highway_Type']]).toarray()
data = Data(x=torch.tensor(X_scaled, dtype=torch.float32), edge_index=edge_index)
weight_gk = adj_matrix_highway["Weight_gk"].values.reshape(-1, 1)
edge_attr = np.hstack((weight_gk, waytype_encoded))
data.edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
data.y = torch.tensor(y, dtype=torch.float32)
# 3.2 Split indices for training, validation, and testing sets
train_val_idx, test_idx = train_test_split(np.arange(data.num_nodes), test_size=0.2, random_state=41)
train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=41)
# 3.3 Create masks for training, validation, and testing
data.train_mask = torch.tensor(np.isin(np.arange(data.num_nodes), train_idx), dtype=torch.bool)
data.val_mask = torch.tensor(np.isin(np.arange(data.num_nodes), val_idx), dtype=torch.bool)
data.test_mask = torch.tensor(np.isin(np.arange(data.num_nodes), test_idx), dtype=torch.bool)
# 3.4 Save processed data and print its contents
data_path = 'data/processed/data_homo.pt'
torch.save(data, data_path)
print(f"Data objects: {data}")
