import pandas as pd
import seaborn as sns

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置Seaborn的绘图风格为白底网格
sns.set_theme(style="whitegrid")

# 加载数据
bike_features = pd.read_csv('../data/inputs/bike_15mins_filtered.csv').drop('LocationID', axis=1)
bus_features = pd.read_csv('../data/inputs/bus_15mins_filtered.csv').drop('LocationID', axis=1)
tube_features = pd.read_csv('../data/inputs/train_15mins_filtered.csv').drop('LocationID', axis=1)

# 计算每个时间点的平均流量和标准差
bike_means = bike_features.mean()
bike_stds = bike_features.std()

bus_means = bus_features.mean()
bus_stds = bus_features.std()

tube_means = tube_features.mean()
tube_stds = tube_features.std()

# 从 06:00 到 22:00，总共64个时间点，每15分钟一个
time_labels = [f"{hour:02d}:00" for hour in range(6, 22)]

# 创建绘图布局
fig, axs = plt.subplots(3, 1, figsize=(9, 9))

# 绘制 Bike 的平均流量曲线和误差条
sns.lineplot(ax=axs[0], x=range(len(bike_means)), y=bike_means, label='Bike', color='#ff7f00')
axs[0].fill_between(range(len(bike_means)), bike_means - bike_stds, bike_means + bike_stds, color='#ff7f00', alpha=0.3)
axs[0].set_title('Bike station average outbound distribution')
axs[0].set_xticks(range(0, 64, 4))
axs[0].set_xticklabels(time_labels, rotation=45)
axs[0].set_ylabel('Average outbound flow')
axs[0].set_ylim(bottom=0)
axs[0].legend()

# 绘制 Bus 的平均流量曲线和误差条
sns.lineplot(ax=axs[1], x=range(len(bus_means)), y=bus_means, label='Bus', color='#33a02c')
axs[1].fill_between(range(len(bus_means)), bus_means - bus_stds, bus_means + bus_stds, color='#33a02c', alpha=0.3)
axs[1].set_title('Bus station average outbound distribution')
axs[1].set_xticks(range(0, 64, 4))
axs[1].set_xticklabels(time_labels, rotation=45)
axs[1].set_ylabel('Average outbound flow')
axs[1].set_ylim(bottom=0)
axs[1].legend()

# 绘制 Tube 的平均流量曲线和误差条
sns.lineplot(ax=axs[2], x=range(len(tube_means)), y=tube_means, label='Tube', color='#1f78b4')
axs[2].fill_between(range(len(tube_means)), tube_means - tube_stds, tube_means + tube_stds, color='#1f78b4', alpha=0.3)
axs[2].set_title('Tube station average outbound distribution')
axs[2].set_xticks(range(0, 64, 4))
axs[2].set_xticklabels(time_labels, rotation=45)
axs[2].set_ylabel('Average outbound flow')
axs[2].set_ylim(bottom=0)
axs[2].legend()

plt.tight_layout()

plt.savefig('results/nodes_traffic.png', dpi=300)