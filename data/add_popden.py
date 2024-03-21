import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
from tqdm import tqdm

path = "./label/"
files = [
    "merged_bikelocation_POI_filtered.csv",
    "merged_buslocation_POI_filtered.csv",
    "merged_trainlocation_POI_filtered.csv"
]
output_files = ["merged_bikelocation_POI_filtered_wr.csv",
    "merged_buslocation_POI_filtered_wr.csv",
    "merged_trainlocation_POI_filtered_wr.csv"]



# 创建点对象
def create_point(easting, northing):
    return Point(easting, northing)


# 读取GIS文件
oa_gdf = gpd.read_file('./map/OA_2011_London_gen_MHW.shp')


# 计算加权人口密度
def calculate_weighted_population_density(station_point, oa_gdf, radius=1000):
    buffer = station_point.buffer(radius)
    relevant_oas = oa_gdf[oa_gdf.intersects(buffer)]

    total_weighted_popden = 0
    total_area = 0
    for _, oa in relevant_oas.iterrows():
        intersection_area = oa['geometry'].intersection(buffer).area
        total_weighted_popden += intersection_area * oa['POPDEN']
        total_area += intersection_area

    if total_area > 0:
        return total_weighted_popden / total_area
    else:
        return None


# 处理每个CSV文件
for file in files:
    df = pd.read_csv(path + file)

    # 添加进度条
    tqdm.pandas(desc=f"Processing {file}")

    df['station_point'] = df.apply(lambda x: create_point(x['easting'], x['northing']), axis=1)
    df['population_density'] = df['station_point'].progress_apply(
        lambda x: calculate_weighted_population_density(x, oa_gdf))

    df.drop(['station_point'], axis=1, inplace=True)
    df.to_csv(path + 'wr_' + file, index=False)