import xml.etree.ElementTree as ET
import geopandas as gpd
from shapely.geometry import Point

# 加载 XML 文件
tree = ET.parse('./bus/Route_Geometry_1_20220318.xml')
root = tree.getroot()

# 准备存储路线数据的列表
data = []

# 遍历每个 Route_Geometry 元素
for route_geometry in root.findall('./Route_Geometry'):
    contract_line_no = route_geometry.get('aContract_Line_No')
    lbsl_run_no = route_geometry.get('aLBSL_Run_No')
    sequence_no = route_geometry.get('aSequence_No')
    direction = route_geometry.find('Direction').text
    longitude = float(route_geometry.find('Location_Longitude').text)
    latitude = float(route_geometry.find('Location_Latitude').text)

    # 创建一个点对象
    point = Point(longitude, latitude)

    # 确保键名为 'geometry'，并且每个字典都包含此键
    data.append({
        "contract_line_no": contract_line_no,
        "lbsl_run_no": lbsl_run_no,
        "sequence_no": sequence_no,
        "direction": direction,
        "geometry": point  # 确保这里使用的是 'geometry'
    })

# 显式创建 GeoDataFrame，并确保传入正确的 geometry 参数
gdf = gpd.GeoDataFrame(data, geometry='geometry', crs="EPSG:4326")

# 输出 GeoDataFrame 查看结果
print(gdf.head())