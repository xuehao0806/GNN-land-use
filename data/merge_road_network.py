import osmnx as ox
import pickle
import networkx as nx

# 确保osmnx使用的是最新的街道网络数据
ox.config(use_cache=True, log_console=True)

# 加载您已经有的大伦敦地区的图G
with open("graph/london_graph.pkl", "rb") as f:
    G = pickle.load(f)

# 下载City of London的街道网络数据。这里使用的是驾车网络，您可以根据需要更改为步行或自行车网络。
G_city_of_london = ox.graph_from_place('City of London, London, United Kingdom', network_type='drive')

# 合并两个图。如果两个图中有相同的节点和边，networkx会自动处理。
G_combined = nx.compose(G, G_city_of_london)

# 定义要保留的highway类型
reserved_highways = {'residential', 'primary', 'secondary', 'tertiary'}

# 遍历图中的每一条边
for u, v, key, data in G_combined.edges(keys=True, data=True):
    # 检查highway类型，如果是列表就取第一个元素
    if isinstance(data['highway'], list):
        data['highway'] = data['highway'][0]

    # 现在data['highway']一定是一个字符串，可以直接判断并替换（如需）
    if data['highway'] not in reserved_highways:
        data['highway'] = 'unclassified'

_, edges = ox.graph_to_gdfs(G)

# 显示边的属性
print("\n边属性：")
print(edges.columns)
# print(edges.head())

highway_counts = edges['highway'].value_counts()
# 打印结果
print("每种highway类型的边的数量：")
print(highway_counts)

output_pickle_path = 'graph/all_london_graph_processed.pkl'
# 使用pickle保存图到文件
with open(output_pickle_path, 'wb') as f:
    pickle.dump(G_combined, f)
print('finished')

