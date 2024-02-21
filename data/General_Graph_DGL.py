import dgl
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm


def read_sort_data(file_path):
    """
    Read and sort data by 'LocationID' from a CSV file.

    Parameters:
    - file_path: str, the path to the CSV file.

    Returns:
    - sorted_data: DataFrame, data sorted by 'LocationID'.
    """
    try:
        # Load the data
        data = pd.read_csv(file_path)
        # Sort the data by 'LocationID'
        sorted_data = data.sort_values(by='LocationID')
        return sorted_data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()


def euclidean_distance(x1, y1, x2, y2):
    """
    Calculate the Euclidean distance between two points in the UK OSNG coordinate system.
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def add_features_and_labels(g, node_type, features, labels):
    """
    Add features and labels to nodes of a specific type in the graph.
    """
    g.nodes[node_type].data['feat'] = torch.tensor(features.values, dtype=torch.float32)
    g.nodes[node_type].data['label'] = torch.tensor(labels.values, dtype=torch.float32)


def build_heterogeneous_graph_from_data(subway_data, bus_data, bike_data, subway_features, bus_features, bike_features, subway_labels, bus_labels, bike_labels, threshold=1000):
    # Create an empty dictionary to hold the graph data
    graph_data = {}

    # Function to check data and add nodes
    def check_and_add_nodes(data, ntype):
        if not data.empty and 'Easting' in data.columns and 'Northing' in data.columns:
            num_nodes = len(data)
            graph_data[(ntype, ntype + '_to_' + ntype, ntype)] = (torch.arange(num_nodes), torch.arange(num_nodes))
            return True
        return False

    # Add nodes for each transportation type if data is valid
    subway_valid = check_and_add_nodes(subway_data, 'subway')
    bus_valid = check_and_add_nodes(bus_data, 'bus')
    bike_valid = check_and_add_nodes(bike_data, 'bike')

    # Function to find edges within a threshold distance
    def find_edges_within_threshold(data1, data2, ntype1, ntype2):
        edges = []
        for i, row1 in tqdm(data1.iterrows(), total=len(data1), desc=f"Processing {ntype1}-{ntype2} Edges"):
            for j, row2 in data2.iterrows():
                dist = np.sqrt((row2['Easting'] - row1['Easting']) ** 2 + (row2['Northing'] - row1['Northing']) ** 2)
                if dist <= threshold:
                    edges.append((i, j))
        if edges:
            graph_data[(ntype1, f'{ntype1}_to_{ntype2}', ntype2)] = tuple(zip(*edges))

    # Add edges based on the valid data
    if subway_valid:
        find_edges_within_threshold(subway_data, subway_data, 'subway', 'subway')
    if bus_valid:
        find_edges_within_threshold(bus_data, bus_data, 'bus', 'bus')
    if bike_valid:
        find_edges_within_threshold(bike_data, bike_data, 'bike', 'bike')
    if subway_valid and bus_valid:
        find_edges_within_threshold(subway_data, bus_data, 'subway', 'bus')
    if subway_valid and bike_valid:
        find_edges_within_threshold(subway_data, bike_data, 'subway', 'bike')
    if bus_valid and bike_valid:
        find_edges_within_threshold(bus_data, bike_data, 'bus', 'bike')

    # Create the graph if there is at least one valid type
    if subway_valid or bus_valid or bike_valid:
        g = dgl.heterograph(graph_data)
    else:
        raise ValueError("No valid data provided for graph construction")

    # After constructing the graph, add features and labels
    if not subway_data.empty:
        add_features_and_labels(g, 'subway', subway_features.iloc[:, 1:], subway_labels.iloc[:, 1:])
    if not bus_data.empty:
        add_features_and_labels(g, 'bus', bus_features.iloc[:, 1:], bus_labels.iloc[:, 1:])
    if not bike_data.empty:
        add_features_and_labels(g, 'bike', bike_features.iloc[:, 1:], bike_labels.iloc[:, 1:])

    return g

print('hh')

feature_num = 64
label_num = 6
bus_data = read_sort_data('./label/merged_buslocation_POI_filtered.csv')
subway_data = read_sort_data('./label/merged_trainlocation_POI_filtered.csv')
bike_data = read_sort_data('./label/merged_bikelocation_POI_filtered.csv')

# Reading and sorting data for features and labels
bike_features = read_sort_data('./inputs/bike_15mins_filtered.csv').iloc[:, -65:]
bus_features = read_sort_data('./inputs/bus_15mins_filtered.csv').iloc[:, -65:]
subway_features = read_sort_data('./inputs/train_15mins_filtered_1.csv').iloc[:, -65:]

bike_labels = read_sort_data('./label/processed_merged_bikelocation_POI_filtered.csv').iloc[:, -7:]
bus_labels = read_sort_data('./label/processed_merged_buslocation_POI_filtered.csv').iloc[:, -7:]
subway_labels = read_sort_data('./label/processed_merged_trainlocation_POI_filtered.csv').iloc[:, -7:]



# 构建图
hetero_graph = build_heterogeneous_graph_from_data(
    subway_data, bus_data, bike_data,
    subway_features, bus_features, bike_features,
    subway_labels, bus_labels, bike_labels
)

# 检查图是否构建正确
# print("Number of nodes for each type:", hetero_graph.num_nodes())
# print("Number of edges for each type:", hetero_graph.num_edges())

print("Features for subway node 0:", hetero_graph.nodes['subway'].data['feat'][0])
print("Label for subway node 0:", hetero_graph.nodes['subway'].data['label'][0])

dgl.save_graphs('./graph/hetero_graph.bin', [hetero_graph])