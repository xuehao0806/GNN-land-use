import pandas as pd

# 读取CSV文件
merged_bikelocation_POI_filtered = pd.read_csv('label/merged_bikelocation_POI_filtered.csv')
merged_buslocation_POI_filtered = pd.read_csv('label/merged_buslocation_POI_filtered.csv')
merged_trainlocation_POI_filtered = pd.read_csv('label/merged_trainlocation_POI_filtered.csv')

# 保留 'LocationID', 'latitude', 'longitude' 列，并为每个DataFrame添加一个新列 'transport_mode'
merged_bikelocation_POI_filtered = merged_bikelocation_POI_filtered[['LocationID', 'latitude', 'longitude']].assign(transport_mode='bike')
merged_buslocation_POI_filtered = merged_buslocation_POI_filtered[['LocationID', 'latitude', 'longitude']].assign(transport_mode='bus')
merged_trainlocation_POI_filtered = merged_trainlocation_POI_filtered[['LocationID', 'latitude', 'longitude']].assign(transport_mode='tube')

# 将所有DataFrame合并为一个
combined_df = pd.concat([merged_bikelocation_POI_filtered, merged_buslocation_POI_filtered, merged_trainlocation_POI_filtered], ignore_index=True)

# 生成 'node_ID' 列，从0开始递增
combined_df['node_ID'] = combined_df.index

# 可选：保存合并后的DataFrame到新的CSV文件
combined_df.to_csv('label/combined_location.csv', index=False)

# 打印前几行以确认结果
print(combined_df.head())

