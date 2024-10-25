import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_violin_for_models(model1_name, loader1_name, model2_name, loader2_name):
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 16})  # 调整全局字体大小

    # 定义一个函数来加载并处理数据
    def load_and_process_data(model_name, loader_name):
        data = pd.read_csv(f'residual/{model_name}_{loader_name}.csv')
        data['mode'] = ['bike'] * 883 + ['bus'] * 3167 + ['tube'] * 186
        data['model'] = data['mode'] + '+' + model_name  # 添加一个列来标识模型
        return data

    # 加载两个模型的数据
    data_model1 = load_and_process_data(model1_name, loader1_name)
    data_model2 = load_and_process_data(model2_name, loader2_name)

    # 合并数据
    data = pd.concat([data_model1, data_model2], ignore_index=True)

    # 设置绘图样式
    sns.set(style="whitegrid")

    # 创建2x3网格布局
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))

    # 定义土地利用指标列表
    land_uses = ['office', 'sustenance', 'transport', 'retail', 'leisure', 'residence']

    # 为每种旅行模式和模型组合配置颜色
    palette = {
        "bike+" + model1_name: "#FF9050", "bike+" + model2_name: "#FF9050",
        "bus+" + model1_name: "#7EC682", "bus+" + model2_name: "#7EC682",
        "tube+" + model1_name: "#7ECAFD", "tube+" + model2_name: "#7ECAFD"
    }

    # 模型组合的期望顺序
    desired_order = [
        "bike+" + model1_name, "bike+" + model2_name,
        "bus+" + model1_name, "bus+" + model2_name,
        "tube+" + model1_name, "tube+" + model2_name
    ]

    # 为每个土地利用指标绘制小提琴图
    for i, land_use in enumerate(land_uses):
        row, col = divmod(i, 3)
        ax = axes[row, col]

        # 过滤数据并绘制小提琴图
        # filtered_data = data[(data[land_use] >= -0.4) & (data[land_use] <= 0.4)]
        violin_plot = sns.violinplot(x='model', y=land_use, data=data, ax=ax, palette=palette, inner='quartile', order=desired_order)
        ax.set_title(f'{land_use.capitalize()}', fontsize=18)  # 调整标题字体大小
        ax.set_ylabel('Residual value', fontsize=16)  # 调整y轴标签字体大小
        # ax.set_xlabel('Model', fontsize=16)  # 调整x轴标签字体大小
        ax.tick_params(axis='x', labelsize=15)  # 调整x轴刻度字体大小
        ax.set_ylim(-0.4, 0.4)  # 设置Y轴范围

        # # 绘制分割线
        # for tick in range(len(desired_order) - 1):
        #     ax.axvline(x=tick + 0.5, color='gray', linestyle='--', linewidth=0.5)

    # 调整布局
    plt.tight_layout()

    # 保存合成图像
    plt.savefig('residual/Error_Combined_Vplots.png', dpi=300)

    # 显示图像
    plt.show()
# 调用函数示例
plot_violin_for_models('GCN', 'Neighbor', 'HGT', 'HGTloader')

