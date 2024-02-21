import pandas as pd
import geopandas as gpd
from pyproj import Proj, transform
import folium
from tqdm import tqdm

# 定义边界样式的函数
def style_function(feature):
    return {
        'fillColor': 'none',  # 不填充颜色
        'color': 'grey',  # 灰色边界线
        'weight': 2,  # 边界线宽度
        'dashArray': '5, 5'  # 虚线样式（线段长度和间隔长度）
    }


# 示例数据文件路径，需要根据实际情况进行调整
shapefile_path = "../data/map/lp-falp-2006-inner-outer-london.shp"
subway_stations_path = "../data/label/merged_trainlocation_POI_filtered.csv"
bus_stations_path = "../data/label/merged_buslocation_POI_filtered.csv"
bike_points_path = "../data/label/merged_bikelocation_POI_filtered.csv"

# 加载shapefile文件
london_boundaries = gpd.read_file(shapefile_path).to_crs(epsg=4326)
inner_boundary = london_boundaries.iloc[0]

# 读取其他数据（这里假设数据为CSV格式，包含经纬度列）
subway_stations = pd.read_csv(subway_stations_path)
bus_stations = pd.read_csv(bus_stations_path)
bike_points = pd.read_csv(bike_points_path)

# 创建Folium地图，以伦敦市中心为中心点
map = folium.Map(location=[51.5074, -0.1278], zoom_start=10)
# , tiles='CartoDB positron'

# 将内伦敦的行政边界添加到地图上，应用自定义样式
folium.GeoJson(
    inner_boundary['geometry'],
    name='geojson',
    style_function=style_function  # 应用自定义样式
).add_to(map)

# 添加地铁站、公交车站和自行车停放点到地图上
# 为每种类型的站点选择一个颜色
color_subway = '#1f78b4'  # 淡蓝色，用于地铁站
color_bus = '#33a02c'    # 亮绿色，用于公交车站
color_bike = '#ff7f00'   # 亮红色，用于自行车停放点

# 添加地铁站（方形标记）
for index, row in tqdm(subway_stations.iterrows(), total=subway_stations.shape[0], desc="Adding Subway Stations"):
    folium.CircleMarker([row['latitude'], row['longitude']],
                                radius=10,
                                color='black',
                                fill=True,
                                fill_color=color_subway,
                                weight=1.5,
                                fill_opacity=0.65).add_to(map)

# 添加公交车站（圆形标记）
for index, row in tqdm(bus_stations.iterrows(), total=bus_stations.shape[0], desc="Adding Bus Stations"):
    folium.CircleMarker([row['latitude'], row['longitude']],
                        radius=5,
                        color='black',
                        fill=True,
                        fill_color=color_bus,
                        weight=1.5,
                        fill_opacity=0.65).add_to(map)

# 添加自行车停放点（三角形标记）
for index, row in tqdm(bike_points.iterrows(), total=bike_points.shape[0], desc="Adding Bike Points"):
    folium.CircleMarker([row['latitude'], row['longitude']],
                                radius=5,
                                color='black',
                                fill=True,
                                fill_color=color_bike,
                                weight=1.5,
                                fill_opacity=0.65).add_to(map)

# 保存为HTML文件
map.save('./results/inner_london_station_map.html')
