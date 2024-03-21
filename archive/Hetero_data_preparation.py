import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


def add_split_masks_hetero(data, node_types, train_ratio, val_ratio):
    """
    使用sklearn的train_test_split函数为不同类型的节点添加随机分布的训练、验证和测试掩码。

    :param data: HeteroData对象，包含图数据。
    :param node_types: 节点类型的列表，例如['bike', 'bus', 'tube']。
    :param train_ratio: 用于训练的数据比例。
    :param val_ratio: 用于验证的数据比例。
    """
    test_ratio = 1 - train_ratio - val_ratio
    assert test_ratio > 0, "Invalid train/val ratio."

    for node_type in node_types:
        node_count = data[node_type].x.size(0)

        # 先分割出测试集
        train_val_indices, test_indices = train_test_split(
            range(node_count), test_size=test_ratio, random_state=42)

        # 再从剩余数据中分割出验证集
        train_ratio_adjusted = train_ratio / (1 - test_ratio)
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_ratio / (train_ratio + val_ratio), random_state=42)

        # 初始化掩码
        data[node_type].train_mask = torch.zeros(node_count, dtype=torch.bool)
        data[node_type].val_mask = torch.zeros(node_count, dtype=torch.bool)
        data[node_type].test_mask = torch.zeros(node_count, dtype=torch.bool)

        # 设置掩码
        data[node_type].train_mask[train_indices] = True
        data[node_type].val_mask[val_indices] = True
        data[node_type].test_mask[test_indices] = True

# 1. 加载节点特征数据
bike_features = pd.read_csv('data/inputs/bike_15mins_filtered.csv')
bus_features = pd.read_csv('data/inputs/bus_15mins_filtered.csv')
tube_features = pd.read_csv('data/inputs/train_15mins_filtered.csv')

# 2. 加载标签数据
label_columns_list = ['office', 'sustenance', 'transport', 'retail', 'leisure', 'residence']
bike_labels = pd.read_csv('data/label/processed_merged_bikelocation_POI_filtered_wr.csv')
bus_labels = pd.read_csv('data/label/processed_merged_buslocation_POI_filtered_wr.csv')
tube_labels = pd.read_csv('data/label/processed_merged_trainlocation_POI_filtered_wr.csv')
# 初始化异构图数据结构
hetero_data = HeteroData()
# 特征标准化
scaler = StandardScaler()
# 3. 为每种交通工具类型构建节点数据
for node_type, features, labels in [('bike', bike_features, bike_labels),
                                    ('bus', bus_features, bus_labels),
                                    ('tube', tube_features, tube_labels)]:

    # 特征标准化处理
    features_scaled = scaler.fit_transform(features.drop('LocationID', axis=1))
    # 合并特征和标签到HeteroData
    hetero_data[node_type].x = torch.tensor(features_scaled, dtype=torch.float32)
    hetero_data[node_type].y = torch.tensor(labels[label_columns_list].values, dtype=torch.float32)

# 加载边表数据
adj_matrix_highway = pd.read_csv("data/edge/real_knn_edges_highway.csv")

# 为不同节点类型之间的所有可能边添加边
for start_type, start_range, end_type, end_range in [
    ('bike', (0, 883), 'bike', (0, 883)),
    ('bike', (0, 883), 'bus', (883, 4050)),
    ('bike', (0, 883), 'tube', (4050, 4236)),
    ('bus', (883, 4050), 'bike', (0, 883)),
    ('bus', (883, 4050), 'bus', (883, 4050)),
    ('bus', (883, 4050), 'tube', (4050, 4236)),
    ('tube', (4050, 4236), 'bike', (0, 883)),
    ('tube', (4050, 4236), 'bus', (883, 4050)),
    ('tube', (4050, 4236), 'tube', (4050, 4236))
]:
    # 根据起始和结束ID范围过滤边
    filtered_edges = adj_matrix_highway[
        (adj_matrix_highway['Start'].between(*start_range)) &
        (adj_matrix_highway['End'].between(*end_range))
        ]

    # 调整ID并创建Tensor
    start_indices = filtered_edges['Start'].apply(lambda x: x - start_range[0]).values
    end_indices = filtered_edges['End'].apply(lambda x: x - end_range[0]).values
    edge_index = torch.tensor(np.array([start_indices, end_indices]), dtype=torch.long)

    hetero_data[start_type, f'{start_type}_to_{end_type}', end_type].edge_index = edge_index

# print(f"HeteroData object: {hetero_data}")
add_split_masks_hetero(hetero_data, ['bike', 'bus', 'tube'], 0.7, 0.15)
print(f"HeteroData object: {hetero_data}")

data_path = 'data/processed/data_hetero.pt'
torch.save(hetero_data, data_path)