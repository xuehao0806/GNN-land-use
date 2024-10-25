import json
import folium
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

# 读取JSON文件
def load_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

filepath = 'tfl/tfl_lines.json'  # JSON文件路径
filepath2 = 'tfl/tfl_stations.json'  # JSON文件路径
shapefile_path = "../data/map/lp-falp-2006-inner-outer-london.shp"
lines_features = load_data(filepath)
stations_features = load_data(filepath2)
# 假设 json_data 是从JSON文件读取的数据
# 处理线路数据
line_df = pd.json_normalize(lines_features['features'])
line_df['geometry'] = line_df['geometry.coordinates'].apply(lambda coords: LineString(coords))
line_gdf = gpd.GeoDataFrame(line_df, geometry='geometry')

# 处理车站数据
station_df = pd.json_normalize(stations_features['features'])
station_df['geometry'] = station_df['geometry.coordinates'].apply(lambda coords: Point(coords))
station_gdf = gpd.GeoDataFrame(station_df, geometry='geometry')

london_boundaries = gpd.read_file(shapefile_path).to_crs(epsg=4326)
inner_boundary = london_boundaries.iloc[0]

# 创建Folium地图
m = folium.Map(location=[51.5074, -0.1278], zoom_start=14, tiles='CartoDB positron')

# Function to define boundary styles
def style_function(feature):
    return {
        'fillColor': 'none',
        'color': 'grey',
        'weight': 3,
        'dashArray': '5, 5'
    }
# Add Inner London's administrative boundary to the map
folium.GeoJson(inner_boundary['geometry'], name='geojson', style_function=style_function).add_to(m)

# 将GeoDataFrame中的每个LineString添加到地图
for _, row in line_gdf.iterrows():
    folium.PolyLine(
        locations=[(lat, lon) for lon, lat in row['geometry'].coords],
        weight=3,
        color='blue'
    ).add_to(m)


# 添加车站
for _, row in station_gdf.iterrows():
    folium.CircleMarker(
        location=[row['geometry'].y, row['geometry'].x],
        radius=4,
        fill=True,
        color='black',
        fill_color='blue',
        fill_opacity=0.5,
    ).add_to(m)

m.save('./tfl/combined_london_map.html')
print('finished')
