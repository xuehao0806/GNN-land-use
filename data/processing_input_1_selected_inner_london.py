import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# 强制使用Shapely而非PyGEOS
os.environ['USE_PYGEOS'] = '0'

def filter_common_locationID(df1, df2):
    """
    筛选两个DataFrame中的共同LocationID，并返回筛选后的DataFrame。
    """
    # 获取两个DataFrame中LocationID列的交集
    locationID_intersection = set(df1['LocationID']).intersection(set(df2['LocationID']))

    # 将交集转换为列表
    locationID_list = list(locationID_intersection)

    # 筛选df1和df2中属于交集的行
    filtered_df1 = df1[df1['LocationID'].isin(locationID_list)]
    filtered_df2 = df2[df2['LocationID'].isin(locationID_list)]

    # 删除具有重复LocationID的行，只保留第一次出现的行
    filtered_df1_unique = filtered_df1.drop_duplicates(subset='LocationID', keep='first')
    filtered_df2_unique = filtered_df2.drop_duplicates(subset='LocationID', keep='first')

    return filtered_df1_unique, filtered_df2_unique

def filter_london_data(region):
    def load_station_data(file_path):
        df = pd.read_csv(file_path, low_memory=False)
        df['geometry'] = df.apply(lambda row: Point(row['Easting'], row['Northing']), axis=1)
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:27700")
        return gdf

    london_boundaries = gpd.read_file('./map/lp-falp-2006-inner-outer-london.shp').to_crs("EPSG:27700")
    inner_london_boundary = london_boundaries.iloc[0]  # 只选择内伦敦的边界

    # 加载车站数据
    bike_stations = load_station_data('./label/merged_bikelocation_POI.csv')
    bus_stations = load_station_data('./label/merged_buslocation_POI.csv')
    train_stations = load_station_data('./label/merged_trainlocation_POI.csv')

    # 确定使用的列名
    columns_to_use_bus = ['LocationID'] + bus_stations.columns[-64:].tolist()

    if region == 'inner':
        bus_stations_filtered = bus_stations[bus_stations.within(inner_london_boundary.geometry)]
        train_stations_filtered = train_stations[train_stations.within(inner_london_boundary.geometry)]
        bike_stations_filtered = bike_stations[bike_stations.within(inner_london_boundary.geometry)]
    else:
        bus_stations_filtered = bus_stations[~bus_stations.within(inner_london_boundary.geometry)]
        train_stations_filtered = train_stations[~train_stations.within(inner_london_boundary.geometry)]
        bike_stations_filtered = bike_stations[~bike_stations.within(inner_london_boundary.geometry)]

    # 读取15分钟数据
    bus_15mins = pd.read_csv('./inputs/bus_15mins.csv')
    train_15mins = pd.read_csv('./inputs/train_15mins.csv')
    bike_15mins = pd.read_csv('./inputs/bike_15mins.csv')

    # 筛选重复的locationID
    duplicate_bus, duplicate_bus2 = filter_common_locationID(bus_15mins, bus_stations_filtered)
    duplicate_train, duplicate_train2 = filter_common_locationID(train_15mins, train_stations_filtered)
    duplicate_bike, duplicate_bike2 = filter_common_locationID(bike_15mins, bike_stations_filtered)

    # 选择 LocationID 和最后64列，并重命名列
    duplicate_bus = duplicate_bus[['LocationID'] + duplicate_bus.columns[-64:].tolist()]
    duplicate_train = duplicate_train[['LocationID'] + duplicate_train.columns[-64:].tolist()]
    duplicate_bike = duplicate_bike[['LocationID'] + duplicate_bike.columns[-64:].tolist()]

    # 重命名列以匹配 duplicate_bus 的列名
    duplicate_train.columns = duplicate_bus.columns
    duplicate_bike.columns = duplicate_bus.columns

    suffix = '_inner' if region == 'inner' else '_outer'

    # 保存筛选后的数据
    duplicate_bus.to_csv(f'./inputs/outer/bus_15mins_filtered{suffix}.csv', index=False)
    duplicate_train.to_csv(f'./inputs/outer/train_15mins_filtered{suffix}.csv', index=False)
    duplicate_bike.to_csv(f'./inputs/outer/bike_15mins_filtered{suffix}.csv', index=False)

    duplicate_bus2.to_csv(f'./label/outer/merged_buslocation_POI_filtered{suffix}.csv', index=False)
    duplicate_train2.to_csv(f'./label/outer/merged_trainlocation_POI_filtered{suffix}.csv', index=False)
    duplicate_bike2.to_csv(f'./label/outer/merged_bikelocation_POI_filtered{suffix}.csv', index=False)

    print('Completed filtering for', region)

filter_london_data('outer')


# import os
# import pandas as pd
# import geopandas as gpd
# from shapely.geometry import Point
#
# # 如果你想强制使用Shapely而非PyGEOS
# os.environ['USE_PYGEOS'] = '0'
#
# # 载入车站位置数据
# def load_station_data(file_path):
#     df = pd.read_csv(file_path, low_memory=False)
#     df['geometry'] = df.apply(lambda row: Point(row['Easting'], row['Northing']), axis=1)
#     gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:27700")
#     return gdf
#
# # 读取内伦敦边界的Shapefile
# london_boundaries = gpd.read_file('./map/lp-falp-2006-inner-outer-london.shp').to_crs("EPSG:27700")
# inner_london_boundary = london_boundaries.iloc[0]  # 只选择内伦敦的边界
#
# # 加载车站数据
# bike_stations = load_station_data('./label/merged_bikelocation_POI.csv')
# bus_stations = load_station_data('./label/merged_buslocation_POI.csv')
# train_stations = load_station_data('./label/merged_trainlocation_POI.csv')
#
# # 筛选位于内伦敦的车站
# bus_stations_inner = bus_stations[bus_stations.within(inner_london_boundary.geometry)]
# train_stations_inner = train_stations[train_stations.within(inner_london_boundary.geometry)]
# bike_stations_inner = bike_stations[bike_stations.within(inner_london_boundary.geometry)]
#
# # 读取15分钟数据
# bus_15mins = pd.read_csv('./inputs/bus_15mins.csv')
# train_15mins = pd.read_csv('./inputs/train_15mins.csv')
# bike_15mins = pd.read_csv('./inputs/bike_15mins.csv')
#
# # 确保 locationID 列是相同的数据类型
# bus_stations_inner['LocationID'] = bus_stations_inner['LocationID'].astype(str)
# train_stations_inner['LocationID'] = train_stations_inner['LocationID'].astype(str)
# bike_stations_inner['LocationID'] = bike_stations_inner['LocationID'].astype(str)
#
# # 对比并筛选重复的locationID
# duplicate_bus = bus_15mins[bus_15mins['LocationID'].isin(bus_stations_inner['LocationID'])]
# duplicate_train = train_15mins[train_15mins['LocationID'].isin(train_stations_inner['LocationID'])]
# duplicate_bike = bike_15mins[bike_15mins['LocationID'].isin(bike_stations_inner['LocationID'])]
#
# duplicate_bus2 = bus_stations_inner[bus_stations_inner['LocationID'].isin(bus_15mins['LocationID'])]
# duplicate_train2 = train_stations_inner[train_stations_inner['LocationID'].isin(train_15mins['LocationID'])]
# duplicate_bike2 = bike_stations_inner[bike_stations_inner['LocationID'].isin(bike_15mins['LocationID'])]
#
# duplicate_bus.to_csv('./inputs/bus_15mins_filtered.csv', index=False)
# duplicate_train.to_csv('./inputs/train_15mins_filtered.csv', index=False)
# duplicate_bike.to_csv('./inputs/bike_15mins_filtered.csv', index=False)
#
# duplicate_bus2.to_csv('./label/merged_buslocation_POI_filtered.csv', index=False)
# duplicate_train2.to_csv('./label/merged_trainlocation_POI_filtered.csv', index=False)
# duplicate_bike2.to_csv('./label/merged_bikelocation_POI_filtered.csv', index=False)
#
