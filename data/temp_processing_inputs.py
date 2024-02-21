import pandas as pd

# 假设已经按照之前的指示读取和排序了subway_features和subway_labels
# 这里只是示例，你需要替换为实际的DataFrame变量

# 示例DataFrame，实际应用中应替换为读取的数据
subway_features = pd.read_csv('./inputs/train_15mins_filtered.csv')
subway_labels = pd.read_csv('./label/processed_merged_trainlocation_POI_filtered.csv')

# 获取两个DataFrame中LocationID列的交集
locationID_intersection = set(subway_features['LocationID']).intersection(set(subway_labels['LocationID']))

# 将交集转换为列表
locationID_list = list(locationID_intersection)

# 筛选subway_features中属于交集的行
filtered_subway_features = subway_features[subway_features['LocationID'].isin(locationID_list)]

# 筛选subway_labels中属于交集的行
filtered_subway_labels = subway_labels[subway_labels['LocationID'].isin(locationID_list)]

# 删除具有重复LocationID的行，只保留第一次出现的行
filtered_subway_features_unique = filtered_subway_features.drop_duplicates(subset='LocationID', keep='first')

# 打印结果以验证
print(f"Original number of rows: {filtered_subway_features.shape[0]}")
print(f"Number of rows after removing duplicates: {filtered_subway_features_unique.shape[0]}")

filtered_subway_features_unique.to_csv('./inputs/train_15mins_filtered_1.csv', index=False)
