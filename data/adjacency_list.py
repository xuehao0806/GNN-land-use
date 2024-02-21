import pandas as pd
import numpy as np

def read_data(file_path):
    """
    Read data from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        return str(e)

def euclidean_distance(x1, y1, x2, y2):
    """
    Calculate the Euclidean distance between two points in the UK OSNG coordinate system.
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def create_edge_list(data1, data2, max_distance=1000):
    """
    Create an edge list for nodes in data1 and data2 where the distance between nodes is less than max_distance.
    """
    edges = []
    for i in range(len(data1)):
        for j in range(len(data2)):
            if data1 is data2 and i >= j:
                # Avoid duplicate edges and self-loops in the same dataset
                continue
            dist = euclidean_distance(data1.iloc[i]['Easting'], data1.iloc[i]['Northing'],
                                      data2.iloc[j]['Easting'], data2.iloc[j]['Northing'])
            if dist <= max_distance:
                # Creating an edge (i, j)
                edges.append((i, j))
    return edges

# Read the data
bus_data = read_data('./label/merged_buslocation_POI_filtered.csv')
train_data = read_data('./label/merged_trainlocation_POI_filtered.csv')
bike_data = read_data('./label/merged_bikelocation_POI_filtered.csv')

# Generate edges within and between datasets
train_edges = create_edge_list(train_data, train_data)
bus_edges = create_edge_list(bus_data, bus_data)
bike_edges = create_edge_list(bike_data, bike_data)
subway_bus_edges = create_edge_list(train_data, bus_data)
subway_bike_edges = create_edge_list(train_data, bike_data)
bus_bike_edges = create_edge_list(bus_data, bike_data)