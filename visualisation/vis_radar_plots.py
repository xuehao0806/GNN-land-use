import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import matplotlib

def plot_radar_chart(data):
    # 设置字体大小
    matplotlib.rcParams['font.size'] = 14

    # 将数据转换为DataFrame
    df = pd.DataFrame(data)

    # 设置角度，确保图表正确分为六等分
    labels = df['Category']
    num_vars = len(labels)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # 闭合雷达图

    # 绘制雷达图
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 颜色列表

    for idx, column in enumerate(df.columns[1:]):  # 跳过'Category'列
        values = df[column].values.flatten().tolist()
        values += values[:1]  # 闭合数值列表
        ax.plot(angles, values, 'o-', linewidth=2, linestyle='solid', label=column, color=colors[idx])
        ax.fill(angles, values, alpha=0)  # 去掉阴影，设置透明度为0

    # 添加类别标签
    for label, angle in zip(labels, angles[:-1]):
        ax.text(angle, 1.03, label, horizontalalignment='center', verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))

    # 去除角度标记
    ax.set_xticks([])  # 不显示角度标记

    labels2 = ['', '', '', '', '', '']
    # 设置类别标签
    ax.set_thetagrids([angle * 180 / pi for angle in angles][:-1], labels2)

    # 设置雷达图的值范围
    ax.set_ylim(0.4, 1.0)

    # Create a separate legend figure
    fig_legend = plt.figure(figsize=(10, 1))  # Adjust the figure size to accommodate the horizontal layout
    plt.figlegend(*ax.get_legend_handles_labels(), loc='upper left', ncol=5)  # Set ncol to 5 for horizontal layout
    fig_legend.savefig('results/radar_models_legend.png', dpi=300, bbox_inches='tight')

    # 保存图像
    plt.savefig('results/radar_models_bus_tube.png', dpi=300, bbox_inches='tight')
    print('finished')
    plt.close()

# 测试数据
# bus
# test_data = {
#     'Category': ['office', 'sustenance', 'transport', 'retail', 'leisure', 'residence'],
#     'HGT': [0.844, 0.881, 0.824, 0.797, 0.742, 0.782],
#     'RGCN': [0.812, 0.86, 0.812, 0.78, 0.716, 0.741],
#     'GAT': [0.829, 0.878, 0.786, 0.725, 0.536, 0.69],
#     'GraphSage': [0.698, 0.791, 0.685, 0.645, 0.468, 0.52],
#     'GCN': [0.721, 0.834, 0.746, 0.666, 0.463, 0.578]
# }

# bus + tube
test_data = {
    'Category': ['office', 'sustenance', 'transport', 'retail', 'leisure', 'residence'],
    'HGT': [0.866, 0.895, 0.856, 0.801, 0.782, 0.79],
    'RGCN': [0.857, 0.889, 0.838, 0.787, 0.752, 0.754],
    'GAT': [0.834, 0.879, 0.817, 0.738, 0.635, 0.675],
    'GraphSage': [0.795, 0.858, 0.799, 0.691, 0.603, 0.631],
    'GCN': [0.827, 0.87, 0.8, 0.729, 0.621, 0.64]
}

# bus + bike
# test_data = {
# 'Category': ['office', 'sustenance', 'transport', 'retail', 'leisure', 'residence'],
# 'HGT': [0.93, 0.947, 0.926, 0.905, 0.85, 0.876],
# 'RGCN': [0.903, 0.927, 0.914, 0.87, 0.847, 0.866],
# 'GAT': [0.902, 0.955, 0.92, 0.846, 0.807, 0.877],
# 'GraphSage': [0.885, 0.927, 0.908, 0.828, 0.743, 0.81],
# 'GCN': [0.871, 0.926, 0.891, 0.818, 0.715, 0.79]
# }

# test_data = {
#     'Category': ['office', 'sustenance', 'transport', 'retail', 'leisure', 'residence'],
#     'HGT': [0.928, 0.952, 0.936, 0.891, 0.861, 0.895],
#     'RGCN': [0.915, 0.939, 0.927, 0.865, 0.844, 0.866],
#     'GAT': [0.904, 0.952, 0.927, 0.851, 0.789, 0.847],
#     'GraphSage': [0.887, 0.938, 0.912, 0.841, 0.773, 0.831],
#     'GCN': [0.881, 0.935, 0.9, 0.812, 0.753, 0.809]
# }

plot_radar_chart(test_data)