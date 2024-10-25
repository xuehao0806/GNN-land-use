import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取 CSV 文件
dataset = 'outer'
method = 'IG'
data = pd.read_csv(f'./feature_attribution/feature_importance_summary_{dataset}_{method}.csv', index_col=0)
data = -data * 100
# 使用 seaborn 创建热图
# 设置 matplotlib 的全局字体大小
plt.rcParams.update({'font.size': 14})  # 调整全局基础字体大小
plt.rcParams['axes.labelsize'] = 15  # 调整坐标轴标签大小
# plt.rcParams['axes.titlesize'] = 16  # 调整标题大小
plt.figure(figsize=(18, 10))
sns.heatmap(data, cmap='RdBu_r', annot=True, fmt=".1f", annot_kws={"size": 8, "rotation": 90}, linewidths=.5)
# plt.title('Feature Importance Heatmap (Scaled by 100)')
# plt.ylabel('Land use indicators')
# plt.xlabel('Time slots in a day')
plt.savefig(f'feature_attribution/feature_importance_heatmap_{dataset}_{method}.png', dpi=300)

