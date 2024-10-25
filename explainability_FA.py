import torch
import torch.nn.functional as F
from torch_geometric.loader import HGTLoader
from torch_geometric.nn import HGTConv, Linear
from utils import evaluation
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.explain import Explainer, CaptumExplainer, GNNExplainer
from torch_geometric.explain import GNNExplainer

data = torch.load('data/processed/outer/data_hetero.pt')
# print(f"Data objects: {data}")
kwargs = {'batch_size': 64, 'num_workers': 0, 'persistent_workers': False}
# Creating heterogeneous graph training, validation, and test loaders
train_loader = HGTLoader(data, num_samples={'node': [512] * 2}, shuffle=True,
                         input_nodes=('node', data['node'].train_mask), **kwargs)
val_loader = HGTLoader(data, num_samples={'node': [512] * 2},
                       input_nodes=('node', data['node'].val_mask), **kwargs)
test_loader = HGTLoader(data, num_samples={'node': [512] * 2},
                        input_nodes=('node', data['node'].test_mask), **kwargs)


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict['node'])

model_name = "HGT"
model = HGT(hidden_channels=128, out_channels=6, num_heads=2, num_layers=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data, model = data.to(device), model.to(device)
model.load_state_dict(torch.load('models/outer/HGT_HGTLoader.pth'))

results = evaluation(model_name, model, test_loader, device)
print(results)


# %% 计算 self and neighbor importance

def get_masked_edge_index(edge_index_dict, self_edges=True):
    new_edge_index_dict = {}
    for edge_type, edge_index in edge_index_dict.items():
        if self_edges:
            # 保留 self-edges
            mask = edge_index[0] == edge_index[1]
        else:
            # 保留 neighbor-edges
            mask = edge_index[0] != edge_index[1]
        new_edge_index_dict[edge_type] = edge_index[:, mask]
    return new_edge_index_dict


# 设置字体大小
plt.rcParams.update({'font.size': 16})

# 循环遍历不同的 target
for target in range(6):
    # 准备存储输出的列表
    output_self_list = []
    output_neighbor_list = []

    # 循环遍历 test_loader 中的所有 batch
    for batch in tqdm(test_loader, desc=f"Processing batches for target {target}"):
        batch = batch.to(device)
        x_dict, edge_index_dict = batch.x_dict, batch.edge_index_dict

        # 获取 self-edge 和 neighbor-edge 的边索引
        edge_index_dict_self = get_masked_edge_index(edge_index_dict, self_edges=True)
        edge_index_dict_neighbor = get_masked_edge_index(edge_index_dict, self_edges=False)

        # 执行模型，计算 self-edge 和 neighbor-edge 的输出
        output_self = model(x_dict, edge_index_dict_self)
        output_neighbor = model(x_dict, edge_index_dict_neighbor)

        # 提取该 batch 中所有节点的 target 结果
        output_self_target = output_self[:, target].detach().cpu().numpy()
        output_neighbor_target = output_neighbor[:, target].detach().cpu().numpy()
        # output_neighbor_target

        # 将结果添加到列表中
        output_self_list.extend(output_self_target)
        output_neighbor_list.extend(output_neighbor_target)

    # 计算self-edge和neighbor-edge的频率计数和bin
    self_counts, self_bins = np.histogram(output_self_list, bins=200)
    neighbor_counts, neighbor_bins = np.histogram(output_neighbor_list, bins=200)

    # 计算频率比例
    self_freq = self_counts / sum(self_counts)
    neighbor_freq = neighbor_counts / sum(neighbor_counts)

    # 绘制self-edge贡献的频率分布
    plt.hist(self_bins[:-1], bins=self_bins, weights=self_freq, alpha=0.6, color='blue',
             label='Self')

    # 绘制neighbor-edge贡献的频率分布
    plt.hist(neighbor_bins[:-1], bins=neighbor_bins, weights=neighbor_freq, alpha=0.6, color='orange',
             label='Neighbor')

    # 计算并标记均值
    self_mean = np.mean(output_self_list)
    neighbor_mean = np.mean(output_neighbor_list)

    plt.axvline(self_mean, color='blue', linestyle='dashed', linewidth=2)
    plt.axvline(neighbor_mean, color='orange', linestyle='dashed', linewidth=2)

    # 根据均值的位置动态调整文字位置
    if self_mean < neighbor_mean:
        plt.text(self_mean - 0.05, 0.65, f'{self_mean:.2f}', color='blue', ha='right')
        plt.text(neighbor_mean + 0.01, 0.65, f'{neighbor_mean:.2f}', color='orange', ha='left')
    else:
        plt.text(self_mean + 0.01, 0.65, f' {self_mean:.2f}', color='blue', ha='left')
        plt.text(neighbor_mean - 0.05, 0.65, f'{neighbor_mean:.2f}', color='orange', ha='right')

    # 添加标签和图例
    plt.xlabel('Feature Importance')
    plt.ylabel('Frequency Rate')
    plt.xlim([-0.1, 0.6])  # 限制x轴范围
    plt.ylim([0, 0.7])  # 设定y轴范围为0到0.7
    plt.legend(loc='upper right')

    # 添加网格
    plt.grid(True)

    # 保存图像
    plt.savefig(f'./visualisation/importance_self-neighbor/target_{target}_feature_importance.png', dpi=300,
                bbox_inches='tight')
    plt.close()

print("All images saved.")

# %% 计算 全局feature importance
# # 准备解释器
# explainer = Explainer(
#     model=model,
#     algorithm=CaptumExplainer('InputXGradient'),
#     explanation_type='model',
#     node_mask_type='attributes',
#     model_config=dict(
#         mode='regression',
#         task_level='node',
#         return_type='raw',
#     ),
# )
#
# # 初始化保存 feature importances 的列表
# feature_importances_list = [[] for _ in range(6)]  # 为每个 target 创建一个列表
#
# # 遍历测试集的每一个样本，并使用 tqdm 显示进度
# total_batches = len(test_loader)  # 获取总批次数以显示进度
# for batch in tqdm(test_loader, total=total_batches, desc="Processing batches"):
#     batch = batch.to(device)
#     x_dict, edge_index_dict = batch.x_dict, batch.edge_index_dict
#     for target in range(6):
#         explanation = explainer(x_dict, edge_index_dict, captum_target=target)
#         feature_importances = explanation["node"].node_mask  # 取得当前样本的 feature importance
#         feature_importances_list[target].append(feature_importances)
#
# # 计算每个 target 的绝对值平均 feature importance
# avg_feature_importances = []
# for feature_list in feature_importances_list:
#     avg_importance = torch.cat(feature_list, dim=0).mean(dim=0)[:64]  # 限制计算前 64 个 features
#     # avg_importance_abs = torch.abs(avg_importance)  # 计算绝对值
#     avg_feature_importances.append(avg_importance.cpu().detach().numpy())
#
# # 创建 DataFrame
# columns = [f"{h:02}:{m:02}" for h in range(6, 22) for m in (0, 15, 30, 45)]
# index = ['office', 'sustenance', 'transport', 'retail', 'leisure', 'residence']
# feature_importance_df = pd.DataFrame(avg_feature_importances, index=index, columns=columns)
#
# # 保存 DataFrame
# dataset = 'outer'
# method = 'IG'
# feature_importance_df.to_csv(f'evaluation/feature_attribution/feature_importance_summary_{dataset}_{method}.csv')
# print(feature_importance_df)