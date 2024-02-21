import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# 如果你想强制使用Shapely而非PyGEOS
os.environ['USE_PYGEOS'] = '0'

# 载入车站位置数据
def load_station_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    df['geometry'] = df.apply(lambda row: Point(row['Easting'], row['Northing']), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:27700")
    return gdf

# 读取内伦敦边界的Shapefile
london_boundaries = gpd.read_file('./map/lp-falp-2006-inner-outer-london.shp').to_crs("EPSG:27700")
inner_london_boundary = london_boundaries.iloc[0]  # 只选择内伦敦的边界

# 加载车站数据
bike_stations = load_station_data('./label/merged_bikelocation_POI.csv')
bus_stations = load_station_data('./label/merged_buslocation_POI.csv')
train_stations = load_station_data('./label/merged_trainlocation_POI.csv')

# 筛选位于内伦敦的车站
bus_stations_inner = bus_stations[bus_stations.within(inner_london_boundary.geometry)]
train_stations_inner = train_stations[train_stations.within(inner_london_boundary.geometry)]
bike_stations_inner = bike_stations[bike_stations.within(inner_london_boundary.geometry)]

# 读取15分钟数据
bus_15mins = pd.read_csv('./inputs/bus_15mins.csv')
train_15mins = pd.read_csv('./inputs/train_15mins.csv')
bike_15mins = pd.read_csv('./inputs/bike_15mins.csv')

# 确保 locationID 列是相同的数据类型
bus_stations_inner['LocationID'] = bus_stations_inner['LocationID'].astype(str)
train_stations_inner['LocationID'] = train_stations_inner['LocationID'].astype(str)
bike_stations_inner['LocationID'] = bike_stations_inner['LocationID'].astype(str)

# 对比并筛选重复的locationID
duplicate_bus = bus_15mins[bus_15mins['LocationID'].isin(bus_stations_inner['LocationID'])]
duplicate_train = train_15mins[train_15mins['LocationID'].isin(train_stations_inner['LocationID'])]
duplicate_bike = bike_15mins[bike_15mins['LocationID'].isin(bike_stations_inner['LocationID'])]

duplicate_bus2 = bus_stations_inner[bus_stations_inner['LocationID'].isin(bus_15mins['LocationID'])]
duplicate_train2 = train_stations_inner[train_stations_inner['LocationID'].isin(train_15mins['LocationID'])]
duplicate_bike2 = bike_stations_inner[bike_stations_inner['LocationID'].isin(bike_15mins['LocationID'])]

duplicate_bus.to_csv('./inputs/bus_15mins_filtered.csv', index=False)
duplicate_train.to_csv('./inputs/train_15mins_filtered.csv', index=False)
duplicate_bike.to_csv('./inputs/bike_15mins_filtered.csv', index=False)

duplicate_bus2.to_csv('./label/merged_buslocation_POI_filtered.csv', index=False)
duplicate_train2.to_csv('./label/merged_trainlocation_POI_filtered.csv', index=False)
duplicate_bike2.to_csv('./label/merged_bikelocation_POI_filtered.csv', index=False)

