import pandas as pd

# 读取CSV文件
merged_bikelocation_POI_filtered = pd.read_csv('label/outer/merged_bikelocation_POI_filtered_outer.csv')
merged_buslocation_POI_filtered = pd.read_csv('label/outer/merged_buslocation_POI_filtered_outer.csv')
merged_trainlocation_POI_filtered = pd.read_csv('label/outer/merged_trainlocation_POI_filtered_outer.csv')
merged_bikelocation_POI_filtered.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True)

# 保留 'LocationID', 'latitude', 'longitude' 列，并为每个DataFrame添加一个新列 'transport_mode'
merged_bikelocation_POI_filtered = merged_bikelocation_POI_filtered[['LocationID', 'latitude', 'longitude']].assign(transport_mode='bike')
merged_buslocation_POI_filtered = merged_buslocation_POI_filtered[['LocationID', 'latitude', 'longitude']].assign(transport_mode='bus')
merged_trainlocation_POI_filtered = merged_trainlocation_POI_filtered[['LocationID', 'latitude', 'longitude']].assign(transport_mode='tube')

# 将所有DataFrame合并为一个
combined_df = pd.concat([merged_bikelocation_POI_filtered, merged_buslocation_POI_filtered, merged_trainlocation_POI_filtered], ignore_index=True)

# 生成 'node_ID' 列，从0开始递增
combined_df['node_ID'] = combined_df.index

# 可选：保存合并后的DataFrame到新的CSV文件
combined_df.to_csv('label/outer/combined_location_outer.csv', index=False)

# 打印前几行以确认结果
print(combined_df.head())

