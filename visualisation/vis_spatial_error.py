import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import seaborn as sns
import matplotlib
import folium
from geopandas import GeoDataFrame
from shapely.geometry import Point
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap

# model_name = 'HGT'
# loader_name = 'HGTLoader'
model_name = 'GCN'
loader_name = 'Neighbor'
# 读取残差
residuals_df = pd.read_csv(f"./residual/{model_name}_{loader_name}.csv")
location_df = pd.read_csv(f"../data/label/combined_location.csv")
df = pd.concat([residuals_df, location_df], axis=1)

# # 读取boroughs文件
# boroughs = gpd.read_file('data/map/London_Borough_Excluding_MHW.shp')

# # 转换坐标系为WGS84
# boroughs = boroughs.to_crs(epsg=4326)

# 根据longitude和latitude创建点坐标
geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
crs = {'init': 'epsg:4326'}

# 将DataFrame转化为GeoDataFrame
geo_df = GeoDataFrame(df, crs=crs, geometry=geometry)


def get_color(residual):
    if residual < 0:
        return 'dodgerblue'
    else:
        return 'orangered'

def size(residual):
    if abs(residual) > 9:
        return 9
    else:
        return abs(residual)


def generate_residual_maps(geo_df, residual_features):
    for feature in residual_features:
        # 创建一个Folium地图对象，设置地图中心和缩放级别
        m = folium.Map(location=[51.5074, -0.1278], zoom_start=13, tiles='CartoDB positron')

        # # 将每个borough添加到地图上
        # for _, r in boroughs.iterrows():
        #     # 对geometry进行检查
        #     if r['geometry'].geom_type == 'Polygon':
        #         geom_list = [r['geometry']]
        #     else:  # 当geom_type为'MultiPolygon'时
        #         geom_list = list(r['geometry'].geoms)
        #
        #     for geom in geom_list:
        #         # 获取每个polygon的边界经纬度，并转换为 (lat, lon) 格式
        #         exterior = [coords[::-1] for coords in geom.exterior.coords]
        #
        #         # 创建并添加polygon
        #         folium.Polygon(exterior, color="grey", fill_opacity=0).add_to(m)

        max_abs_residual = geo_df[feature].abs().max()
        min_abs_residual = geo_df[feature].abs().min()
        norm_radius = Normalize(vmin=min_abs_residual, vmax=max_abs_residual)

        for idx, row in geo_df.iterrows():
            folium.CircleMarker([row['latitude'], row['longitude']],
                                radius=3 + 30 * size(row[feature]),  # 使用绝对值的残差来调整大小
                                fill=True,
                                fill_color=get_color(row[feature]),
                                fill_opacity=0.35,
                                color='black',
                                weight= 1.5,
                                popup=row[feature]
                                ).add_to(m)
        m.save(f'./residual/{feature}_{model_name}_map.html')
        print(f"complete map with {feature}")

# 残差特征列表
residual_features = ['office', 'sustenance', 'transport', 'retail', 'leisure', 'residence']

# 为每一个残差特征生成地图
generate_residual_maps(geo_df, residual_features)