import pandas as pd
import geopandas as gpd
from pyproj import Proj, transform
import folium
from tqdm import tqdm

# Function to define boundary styles
def style_function(feature):
    return {
        'fillColor': 'none',  # No fill color
        'color': 'grey',  # Grey boundary line
        'weight': 2,  # Boundary line width
        'dashArray': '5, 5'  # Dashed line style (segment length and gap length)
    }

# Sample data file paths, adjust according to actual situation
shapefile_path = "../data/map/lp-falp-2006-inner-outer-london.shp"
subway_stations_path = "../data/label/merged_trainlocation_POI_filtered.csv"
bus_stations_path = "../data/label/merged_buslocation_POI_filtered.csv"
bike_points_path = "../data/label/merged_bikelocation_POI_filtered.csv"

# Load shapefile
london_boundaries = gpd.read_file(shapefile_path).to_crs(epsg=4326)
inner_boundary = london_boundaries.iloc[0]

# Read other data (assuming data is in CSV format, containing latitude and longitude columns)
subway_stations = pd.read_csv(subway_stations_path)
bus_stations = pd.read_csv(bus_stations_path)
bike_points = pd.read_csv(bike_points_path)

# Create a Folium map centered at London's city center
map = folium.Map(location=[51.5074, -0.1278], zoom_start=15)
# , tiles='CartoDB positron'

# Add Inner London's administrative boundary to the map using the custom style
folium.GeoJson(
    inner_boundary['geometry'],
    name='geojson',
    style_function=style_function  # Apply custom style
).add_to(map)

# Add subway stations, bus stations, and bike points to the map
# Choose a color for each type of station
color_subway = '#1f78b4'  # Light blue for subway stations
color_bus = '#33a02c'    # Bright green for bus stations
color_bike = '#ff7f00'   # Bright orange for bike points

# Add subway stations (square markers)
for index, row in tqdm(subway_stations.iterrows(), total=subway_stations.shape[0], desc="Adding Subway Stations"):
    folium.CircleMarker([row['latitude'], row['longitude']],
                                radius=10,
                                color='black',
                                fill=True,
                                fill_color=color_subway,
                                weight=1.5,
                                fill_opacity=0.65).add_to(map)

# Add bus stations (circle markers)
for index, row in tqdm(bus_stations.iterrows(), total=bus_stations.shape[0], desc="Adding Bus Stations"):
    folium.CircleMarker([row['latitude'], row['longitude']],
                        radius=8,
                        color='black',
                        fill=True,
                        fill_color=color_bus,
                        weight=1.5,
                        fill_opacity=0.65).add_to(map)

# Add bike points (triangle markers)
for index, row in tqdm(bike_points.iterrows(), total=bike_points.shape[0], desc="Adding Bike Points"):
    folium.CircleMarker([row['latitude'], row['longitude']],
                                radius=5,
                                color='black',
                                fill=True,
                                fill_color=color_bike,
                                weight=1.5,
                                fill_opacity=0.65).add_to(map)

# Save as an HTML file
map.save('./results/inner_london_station_map.html')
