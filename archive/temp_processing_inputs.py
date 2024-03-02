import pandas as pd

# 假设已经按照之前的指示读取和排序了subway_features和subway_labels
# 这里只是示例，你需要替换为实际的DataFrame变量

# 示例DataFrame，实际应用中应替换为读取的数据
subway_features = pd.read_csv('./inputs/train_15mins_filtered.csv')
subway_labels = pd.read_csv('./label/processed_merged_trainlocation_POI_filtered.csv')

# 使用merge函数根据'LocationID'同步subway_features和subway_labels
# 注意：这里假设'LocationID'列在两个DataFrame中都存在
synchronized_subway_features = pd.merge(subway_features, subway_labels[['LocationID']], on='LocationID', how='inner')

# 现在synchronized_subway_features只包含在subway_labels中也有的LocationID对应的行

# 你可能想要保留synchronized_subway_features中除了'LocationID'以外的所有列
# synchronized_subway_features = synchronized_subway_features.drop(columns=['LocationID'])

# 如果需要，可以保存结果到新的CSV文件
synchronized_subway_features.to_csv('./inputs/train_15mins_filtered_1.csv', index=False)

# 打印结果查看
print(synchronized_subway_features.head())

# 注意：在实际应用中，你需要替换示例DataFrame为实际的subway_features和subway_labels DataFrame变量