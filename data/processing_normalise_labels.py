import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 文件列表

path = "./label/"
files = [
    "merged_bikelocation_POI_filtered_wr.csv",
    "merged_buslocation_POI_filtered_wr.csv",
    "merged_trainlocation_POI_filtered_wr.csv"
]

# 分组和对应列名的字典
groups = {
    "office": [
        "office_government_count", "office_association_count", "office_company_count",
        "office_consulting_count", "office_financial_count", "office_coworking_count",
        "office_lawyer_count", "office_estate_agent_count", "office_insurance_count",
        "office_telecommunication_count"
    ],
    "sustenance": [
        "amenity_restaurant_count", "amenity_cafe_count", "amenity_bar_count",
        "amenity_pub_count", "amenity_fast_food_count", "amenity_bicycle_parking_count"
    ],
    "transport": [
        "amenity_bicycle_parking_count", "railway_subway_entrance_count",
        "public_transport_platform_count", "amenity_bus_station_count",
        "amenity_taxi_count", "amenity_parking_count", "amenity_ferry_terminal_count",
        "highway_bus_stop_count", "amenity_parking_space_count", "railway_tram_stop_count"
    ],
    "retail": [
        "shop_convenience_count", "shop_supermarket_count", "shop_mall_count",
        "shop_department_store_count", "shop_bakery_count", "shop_butcher_count",
        "shop_clothes_count", "shop_hardware_count", "shop_furniture_count",
        "shop_electronics_count"
    ],
    "leisure": [
        "leisure_park_count", "leisure_sports_centre_count", "leisure_playground_count",
        "leisure_stadium_count", "leisure_swimming_pool_count", "leisure_pitch_count",
        "leisure_track_count", "leisure_fitness_centre_count", "leisure_garden_count",
        "leisure_nature_reserve_count"
    ],
    "residence": [
        "population_density"
    ]
}

# 读取并预处理每个文件
for file in files:
    df = pd.read_csv(path + file)

    # 标准化处理需要处理的列
    columns_to_scale = ["office_government_count", "office_association_count", "office_company_count",
        "office_consulting_count", "office_financial_count", "office_coworking_count",
        "office_lawyer_count", "office_estate_agent_count", "office_insurance_count",
        "office_telecommunication_count", "amenity_restaurant_count", "amenity_cafe_count", "amenity_bar_count",
        "amenity_pub_count", "amenity_fast_food_count", "amenity_bicycle_parking_count", "amenity_bicycle_parking_count",
        "railway_subway_entrance_count", "public_transport_platform_count", "amenity_bus_station_count",
        "amenity_taxi_count", "amenity_parking_count", "amenity_ferry_terminal_count",
        "highway_bus_stop_count", "amenity_parking_space_count", "railway_tram_stop_count", "shop_convenience_count",
        "shop_supermarket_count", "shop_mall_count", "shop_department_store_count", "shop_bakery_count", "shop_butcher_count",
        "shop_clothes_count", "shop_hardware_count", "shop_furniture_count", "shop_electronics_count", "leisure_park_count",
        "leisure_sports_centre_count", "leisure_playground_count", "leisure_stadium_count", "leisure_swimming_pool_count",
        "leisure_pitch_count", "leisure_track_count", "leisure_fitness_centre_count", "leisure_garden_count",
        "leisure_nature_reserve_count", "population_density"]
    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    # 构建新表
    new_df = pd.DataFrame(df['LocationID'])

    # 为每个分组计算均值，并进行再次标准化处理
    scaler_group = StandardScaler()
    for group_name, columns in groups.items():
        group_means = df[columns].mean(axis=1).to_frame(name=group_name)
        new_df[group_name] = scaler_group.fit_transform(group_means)

    # 保存新表到CSV
    new_file_name = path + "zscore_processed_" + file
    new_df.to_csv(new_file_name, index=False)

    print(f"Processed and saved {new_file_name}")
