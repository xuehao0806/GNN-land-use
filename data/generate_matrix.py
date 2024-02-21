import pandas as pd
import osmnx as ox
import networkx as nx
import pickle
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm

# 加载城市道路网络
with open("graph/london_graph.pkl", "rb") as f:
    G = pickle.load(f)

# 加载节点坐标和ID
locations = pd.read_csv("label/combined_location.csv")
# 计算每个位置最近的图节点
locations['nearest_node'] = ox.distance.nearest_nodes(G, X=locations['longitude'], Y=locations['latitude'])

# 准备节点坐标数组
coords = np.array([[G.nodes[node]['x'], G.nodes[node]['y']] for node in locations['nearest_node']])

# 初始化NearestNeighbors找到每个节点最近的5个节点
tree = NearestNeighbors(n_neighbors=11, algorithm='auto', metric='haversine').fit(np.radians(coords))
distances, indices = tree.kneighbors(np.radians(coords))

edges = []
# 进度条显示
for i in tqdm(range(len(locations)), desc="Calculating shortest paths"):
    start_node_id = locations.iloc[i]['node_ID']  # 使用combined_location.csv中的node_ID
    start_node = locations.iloc[i]['nearest_node']  # 图中的最近节点
    for j in range(1, 11):  # 跳过第一个最近节点，因为它是节点本身
        end_node_id = locations.iloc[indices[i, j]]['node_ID']  # 目标节点的ID
        end_node = locations.iloc[indices[i, j]]['nearest_node']  # 图中的最近节点
        try:
            # 计算最短路径长度
            length = nx.shortest_path_length(G, start_node, end_node, weight='length')
            edges.append({'Start': start_node_id, 'End': end_node_id, 'Weight': length})
        except nx.NetworkXNoPath:
            continue  # 如果没有路径，则跳过

# 将边列表转换为DataFrame
edges_df = pd.DataFrame(edges)

edges_df['Weight'] = edges_df['Weight'] / 1000
sigma = 0.5
edges_df['Weight_gk'] = np.exp(-np.square(edges_df['Weight']) / (2 * sigma**2))
edges_df.to_csv("edge/real_knn_edges.csv", index=False)