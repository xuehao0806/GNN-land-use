# # %%
# import os
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # 设置文件路径
# folder_path = 'optimizer/Outer_london'
# # folder_path = 'hidden_size/Outer_london'
# # folder_path = 'sampling_ratio/Outer_london'
# # folder_path = 'noise_rate'
# # 初始化字典来存储每个模型的RMSE均值
# rmse_means = {
#     'GraphSAGE': [],
#     'GCN': [],
#     'GAT': [],
#     # 'NN': [],
#     'RGCN': [],
#     # 'HGT': []
# }
# # 初始化hidden_size
#
# optimizer_list = ['Adam', 'AdamW', 'RMSprop', 'SGD']
# # hidden_sizes = [32, 64, 128, 256, 512]
#
# # # 初始化采样比例
# # sampling_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#
# # 遍历每个模型的csv文件
# for model in rmse_means.keys():
#     for optimizer in optimizer_list:
#         file_name = f'{model}_Neighbor_{optimizer}.csv'
#         file_path = os.path.join(folder_path, file_name)
#
#         # 读取csv文件
#         df = pd.read_csv(file_path)
#
#         # 计算RMSE列的均值并存储
#         rmse_mean = df['RMSE'].mean()
#         rmse_means[model].append(rmse_mean)
#
# # 转换为DataFrame, index 设为hidden_size
# rmse_means_df = pd.DataFrame(rmse_means, index=optimizer_list)
# rmse_means_df.to_csv('optimizer/Outer_london/rmse_means_all.csv', index=True)
# %%
# 保存为汇总CSV文档
# rmse_means_df.to_csv('noise_rate/rmse_means_all.csv', index=False)

# rmse_means_df = pd.read_csv('sampling_ratio/rmse_means_all.csv')
#
# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rmse_means_df = pd.read_csv('optimizer/Outer_london/rmse_means_all.csv')
sns.set(style="whitegrid")

optimizer_list = ['Adam', 'AdamW', 'RMSprop', 'SGD']
# hidden_sizes = pd.Series([32, 64, 128, 256, 512])
# # 创建一个包含0.1到1.0的Series
# sampling_ratios = pd.Series([0.1 * i for i in range(1, 11)])

# Plotting with automatically adjusted y-axis scale based on the new data
plt.figure(figsize=(10, 6))

# Plot each model with circular markers and consistent colors
for model, color in zip(rmse_means_df.columns, sns.color_palette("husl", 5)):
    plt.plot(optimizer_list, rmse_means_df[model], marker='o', color=color, label=model)

# Set the labels with larger font size
plt.xlabel("optimizer_name", fontsize=18)
plt.ylabel("RMSE", fontsize=18)

# Adjust tick parameters for larger font size
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Allow y-axis limit to adjust automatically
plt.ylim(None)

# Modify the legend to increase font size
plt.legend(fontsize=16)

# Show the plot
plt.show()


