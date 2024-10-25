import folium
import pickle
import osmnx as ox
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

# Load the previously saved graph
with open("../../data/graph/all_london_graph_processed.pkl", "rb") as f:
    G = pickle.load(f)

# Create a Folium map centered at London's city center
map = folium.Map(location=[51.5074, -0.1278], zoom_start=14, tiles='CartoDB positron')

# Function to set line width based on the highway type
def get_weight(highway_type):
    if highway_type in ['trunk', 'primary']:
        return 4
    elif highway_type in ['secondary', 'tertiary']:
        return 3
    elif highway_type == 'residential':
        return 2
    else:
        return 2  # Default for 'unclassified' and other cases

# Iterate over each edge in the graph and add it to the map
edges = list(G.edges(data=True))  # Convert edges to a list for tqdm progress
for u, v, data in tqdm(edges, desc='Adding Edges'):
    highway_type = data.get('highway', 'unclassified')  # Default to 'unclassified' if no data
    weight = get_weight(highway_type)
    start_point = (G.nodes[u]['y'], G.nodes[u]['x'])
    end_point = (G.nodes[v]['y'], G.nodes[v]['x'])
    line = folium.PolyLine(locations=[start_point, end_point], weight=weight, color='#cccccc')
    line.add_to(map)

# Function to define boundary styles
def style_function(feature):
    return {
        'fillColor': 'none',
        'color': 'grey',
        'weight': 2,
        'dashArray': '5, 5'
    }

# Load additional data
shapefile_path = "../../data/map/lp-falp-2006-inner-outer-london.shp"
subway_stations_path = "../../data/label/merged_trainlocation_POI_filtered.csv"
bus_stations_path = "../../data/label/merged_buslocation_POI_filtered.csv"
bike_points_path = "../../data/label/merged_bikelocation_POI_filtered.csv"
london_boundaries = gpd.read_file(shapefile_path).to_crs(epsg=4326)
inner_boundary = london_boundaries.iloc[0]
subway_stations = pd.read_csv(subway_stations_path)
bus_stations = pd.read_csv(bus_stations_path)
bike_points = pd.read_csv(bike_points_path)

# Add Inner London's administrative boundary to the map
folium.GeoJson(inner_boundary['geometry'], name='geojson', style_function=style_function).add_to(map)

# Define colors for different station types and add them to the map
color_subway = '#1976D2'  # Brighter blue for subway stations
color_bus = '#388E3C'     # Brighter green for bus stations
color_bike = '#F57C00'    # Brighter orange for bike points

for index, row in tqdm(subway_stations.iterrows(), total=subway_stations.shape[0], desc="Adding Subway Stations"):
    folium.CircleMarker([row['latitude'], row['longitude']], radius=10, color='black',
                        fill=True, fill_color=color_subway, weight=1.5, fill_opacity=0.65).add_to(map)

for index, row in tqdm(bus_stations.iterrows(), total=bus_stations.shape[0], desc="Adding Bus Stations"):
    folium.CircleMarker([row['latitude'], row['longitude']], radius=6, color='black',
                        fill=True, fill_color=color_bus, weight=1.5, fill_opacity=0.65).add_to(map)

for index, row in tqdm(bike_points.iterrows(), total=bike_points.shape[0], desc="Adding Bike Points"):
    folium.CircleMarker([row['latitude'], row['longitude']], radius=4, color='black',
                        fill=True, fill_color=color_bike, weight=1.5, fill_opacity=0.65).add_to(map)

# Save the combined map as an HTML file
map.save('./results/combined_london_map.html')
print('Finished generating the map.')