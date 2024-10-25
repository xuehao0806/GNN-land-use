import folium
import pickle
import osmnx as ox
from tqdm import tqdm

# Load the previously saved graph
with open("../data/graph/all_london_graph_processed.pkl", "rb") as f:
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

# Use tqdm to show progress
# edges = tqdm(G.edges(data=True), desc='Adding Edges')

# Iterate over each edge in the graph and add it to the map
edges = list(G.edges(data=True))  # Convert edges to a list for tqdm progress
for u, v, data in tqdm(edges, desc='Adding Edges'):
    highway_type = data.get('highway', 'unclassified')  # Default to 'unclassified' if no data
    weight = get_weight(highway_type)

    # Get the latitude and longitude for the start and end points
    start_point = (G.nodes[u]['y'], G.nodes[u]['x'])
    end_point = (G.nodes[v]['y'], G.nodes[v]['x'])

    # Create a line representing the road and add it to the map
    line = folium.PolyLine(locations=[start_point, end_point], weight=weight, color='gray')
    line.add_to(map)

# Save the map as an HTML file
map.save('./results/london_road_network.html')
print('finished')