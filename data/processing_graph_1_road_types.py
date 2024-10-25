import pandas as pd
import osmnx as ox
import networkx as nx
import pickle
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm

# 加载城市道路网络
with open("graph/all_london_graph_processed.pkl", "rb") as f:
    G = pickle.load(f)

# 加载节点坐标和ID
# locations = pd.read_csv("label/inner/combined_location.csv")
locations = pd.read_csv("label/outer/combined_location_outer.csv")
# 计算每个位置最近的图节点
locations['nearest_node'] = ox.distance.nearest_nodes(G, X=locations['longitude'], Y=locations['latitude'])

# 准备节点坐标数组
coords = np.array([[G.nodes[node]['x'], G.nodes[node]['y']] for node in locations['nearest_node']])

# 初始化NearestNeighbors找到每个节点最近的10个节点
tree = NearestNeighbors(n_neighbors=11, algorithm='auto', metric='haversine').fit(np.radians(coords))
distances, indices = tree.kneighbors(np.radians(coords))

edges = []
# 进度条显示
for i in tqdm(range(len(locations)), desc="Calculating shortest paths"):
    start_node_id = locations.iloc[i]['node_ID']
    start_node = locations.iloc[i]['nearest_node']
    for j in range(1, 11):  # 跳过第一个最近节点，因为它是节点本身
        end_node_id = locations.iloc[indices[i, j]]['node_ID']
        end_node = locations.iloc[indices[i, j]]['nearest_node']
        try:
            # 计算最短路径
            shortest_path = nx.shortest_path(G, start_node, end_node, weight='length')
            # 初始化道路类型长度的字典
            highway_lengths = {}
            for k in range(len(shortest_path) - 1):
                edge_data = G.get_edge_data(shortest_path[k], shortest_path[k + 1])
                # 保证highway数据不是列表
                highway_type = edge_data[0]['highway'] if isinstance(edge_data[0]['highway'], list) else edge_data[0]['highway']
                length = edge_data[0]['length']
                if highway_type in highway_lengths:
                    highway_lengths[highway_type] += length
                else:
                    highway_lengths[highway_type] = length

            # 如果字典不为空，找出长度最长的道路类型
            if highway_lengths:
                max_highway_type = max(highway_lengths, key=highway_lengths.get)
            else:
                max_highway_type = 'unclassified'

            length = nx.shortest_path_length(G, start_node, end_node, weight='length')
            edges.append({'Start': start_node_id, 'End': end_node_id, 'Weight': length, 'Highway_Type': max_highway_type})
        except nx.NetworkXNoPath:
            continue  # 如果没有路径，则跳过

# 将边列表转换为DataFrame
edges_df = pd.DataFrame(edges)

edges_df['Weight'] = edges_df['Weight'] / 1000
sigma = 0.5
edges_df['Weight_gk'] = np.exp(-np.square(edges_df['Weight']) / (2 * sigma**2))
# edges_df.to_csv("edge/real_knn_edges_highway.csv", index=False)
edges_df.to_csv("edge/real_knn_edges_highway_outer.csv", index=False)