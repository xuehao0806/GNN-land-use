import osmnx as ox
import pickle
import networkx as nx

# Ensure osmnx is using the latest street network data
ox.config(use_cache=True, log_console=True)

# Load the existing graph of Greater London
with open("graph/london_graph.pkl", "rb") as f:
    G = pickle.load(f)

# Download the street network data for the City of London.
# Here we are using the driving network, but you can change it to walking or cycling network if needed.
G_city_of_london = ox.graph_from_place('City of London, London, United Kingdom', network_type='drive')

# Merge the two graphs. If there are the same nodes and edges in both graphs,
# networkx will handle them automatically.
G_combined = nx.compose(G, G_city_of_london)

# Define the highway types to be retained
reserved_highways = {'residential', 'primary', 'secondary', 'tertiary'}

# Iterate through each edge in the graph
for u, v, key, data in G_combined.edges(keys=True, data=True):
    # Check the highway type, and if it is a list, take the first element
    if isinstance(data['highway'], list):
        data['highway'] = data['highway'][0]

    # Now data['highway'] must be a string, so we can directly check and replace it if needed
    if data['highway'] not in reserved_highways:
        data['highway'] = 'unclassified'

_, edges = ox.graph_to_gdfs(G)

# Display edge attributes
print("\nEdge attributes:")
print(edges.columns)
# print(edges.head())

highway_counts = edges['highway'].value_counts()
# Print the results
print("Number of edges for each highway type:")
print(highway_counts)

output_pickle_path = 'graph/all_london_graph_processed.pkl'
# Save the graph to a file using pickle
with open(output_pickle_path, 'wb') as f:
    pickle.dump(G_combined, f)
print('finished')

