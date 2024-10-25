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
from folium.plugins import HeatMap
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata

shapefile_path = "../data/map/lp-falp-2006-inner-outer-london.shp"
london_boundaries = gpd.read_file(shapefile_path).to_crs(epsg=4326)
inner_boundary = london_boundaries.iloc[0]

# model_name = 'HGT'
# loader_name = 'HGTloader'
model_name = 'GCN'
loader_name = 'Neighbor'
# 读取残差
residuals_df = pd.read_csv(f"./residual/{model_name}_{loader_name}.csv")
location_df = pd.read_csv(f"../data/label/combined_location.csv")
df = pd.concat([residuals_df, location_df], axis=1)

# 根据longitude和latitude创建点坐标
geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
crs = {'init': 'epsg:4326'}

# 将DataFrame转化为GeoDataFrame
geo_df = GeoDataFrame(df, crs=crs, geometry=geometry)

# 设置网格分辨率
grid_x, grid_y = np.mgrid[min(df['longitude']):max(df['longitude']):1000j, min(df['latitude']):max(df['latitude']):1000j]

# 预处理数据
points = np.array([df['longitude'], df['latitude']]).T
values = df[['office', 'sustenance', 'transport', 'retail', 'leisure', 'residence']]

# 存储插值结果
interpolated_data = {}

# 对每种用地类型的误差进行插值
for column in values.columns:
    interpolated_data[column] = griddata(points, df[column], (grid_x, grid_y), method='cubic')

# 创建基础地图
map = folium.Map(location=[51.5074, -0.1278], zoom_start=13, tiles='CartoDB positron')

# 准备插值数据用于绘制
def prepare_heatmap_data(interpolated, grid_x, grid_y):
    data = []
    for (i, j), value in np.ndenumerate(interpolated):
        if np.isnan(value):  # 排除NaN值
            continue
        data.append([grid_y[i, j], grid_x[i, j], value])
    return data

# 定义边界样式
def style_function(feature):
    return {
        'fillColor': '#ffffff',
        'color': 'blue',
        'weight': 2,
        'dashArray': '5, 5',
        'fillOpacity': 0.2,
    }

# 添加伦敦内部边界
folium.GeoJson(
    inner_boundary['geometry'],
    name='Inner London Boundary',
    style_function=style_function,
    tooltip='Inner London'
).add_to(map)

# Add Inner London's administrative boundary to the map
folium.GeoJson(inner_boundary['geometry'], name='geojson', style_function=style_function).add_to(map)

# 颜色渐变，从浅蓝到浅红
color_scale = cm.get_cmap('RdBu')
norm = Normalize(vmin=np.nanmin(interpolated_data['office']), vmax=np.nanmax(interpolated_data['office']))

# 为“office”用地类型添加热力图层
heatmap_data = prepare_heatmap_data(interpolated_data['office'], grid_x, grid_y)
HeatMap(data=heatmap_data, gradient={0: '#00f', 1: '#f00'}, min_opacity=0.5, max_val=norm.vmax, radius=15, blur=10).add_to(map)

# 保存并显示地图
map.save('residual/office_inner_London_error_map.html')