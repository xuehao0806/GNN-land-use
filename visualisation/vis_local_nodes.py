import pandas as pd
import folium
# 定义颜色映射
transport_color = {
    'bike': '#ff7f00',
    'bus': '#33a02c',
    'tube': '#1f78b4'
}

# 加载数据
nodes = pd.read_csv('../data/label/combined_location.csv')
edges = pd.read_csv('../data/edge/real_knn_edges_highway.csv')

# 设定需要突出显示的节点
highlight_nodes = [3175, 1177]

# 找出与highlight_nodes相连的所有节点及边
connected_nodes = set(highlight_nodes)
connected_edges = pd.DataFrame(columns=edges.columns)  # 创建一个空的DataFrame来存储相关的边
# 找出与highlight_nodes相连的所有节点
connected_nodes = set(highlight_nodes)
for node in highlight_nodes:
    # 获取以node为起点或终点的边
    edges_from_node = edges[(edges['Start'] == node) | (edges['End'] == node)]
    connected_edges = pd.concat([connected_edges, edges_from_node])

    # 添加节点到集合
    connected_nodes.update(edges_from_node['Start'].tolist())
    connected_nodes.update(edges_from_node['End'].tolist())

# 过滤出这些节点的信息
connected_nodes_data = nodes[nodes['node_ID'].isin(connected_nodes)]

# 创建地图，初始位置设在伦敦中心，缩放级别适当调整
map = folium.Map(location=[51.5074, -0.1278], zoom_start=12)

# 在地图上添加所有相关节点
for index, row in connected_nodes_data.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=7 if row['node_ID'] in highlight_nodes else 5,  # 突出节点更大
        color=transport_color.get(row['transport_mode'], 'gray'),  # 默认颜色为灰色
        fill=True,
        fill_color=transport_color.get(row['transport_mode'], 'gray'),
        popup=f"Node {row['node_ID']} ({row['transport_mode']})",
        ).add_to(map)

# 连接节点
for index, row in connected_edges.iterrows():
    source_node = nodes[nodes['node_ID'] == row['Start']].iloc[0]
    target_node = nodes[nodes['node_ID'] == row['End']].iloc[0]
    folium.PolyLine(
        locations=[
            [source_node['latitude'], source_node['longitude']],
            [target_node['latitude'], target_node['longitude']]
        ],
        color='black',  # 设置线条颜色为黑色
        dash_array='5, 5'  # 设置虚线样式
    ).add_to(map)

# 保存或显示地图
map.save('./results/local_CE.html')  # 保存地图到文件
