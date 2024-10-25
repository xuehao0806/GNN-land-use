import torch
import torch.nn.functional as F
from torch_geometric.loader import HGTLoader
from torch_geometric.nn import HGTConv, Linear
from utils import evaluation
import pandas as pd

data = torch.load('data/processed/data_hetero.pt')
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
model = HGT(hidden_channels=64, out_channels=6, num_heads=2, num_layers=2)
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
data, model = data.to(device), model.to(device)
model.load_state_dict(torch.load('./models/HGT_HGTLoader.pth'))
results = evaluation(model_name, model, test_loader, device)
# print(results)

# get all edge index
data_edge_index = None
prev_edge_index = None
# store edge types according to edge index
edge_type_store = {}
for i, key in enumerate(data.edge_index_dict.keys()):
    data_edge_index = data.edge_index_dict[key]
    for item in data.edge_index_dict[key].t():
        edge_type_store[f"{item[0]},{item[1]}"] = i
    if prev_edge_index is not None:
        data_edge_index = torch.cat((data_edge_index, prev_edge_index), dim=1)
    prev_edge_index = data_edge_index
print(data_edge_index.shape)

# # get data_x
data_x = data.x_dict['node']
# print(data_x.shape)
data_x_node_feat = data_x[:, :64]
print(data_x_node_feat.shape)
data_x_node_type = data_x[:, 64:]
import numpy as np
edge_t0 = torch.unique(data_x_node_type, dim=0)[0]
edge_t1 = torch.unique(data_x_node_type, dim=0)[1]
edge_t2 = torch.unique(data_x_node_type, dim=0)[2]
data_x_node_type_transformed = []
for item in data_x_node_type:
    if (item==edge_t0).all():
        data_x_node_type_transformed.append(0)
    if (item==edge_t1).all():
        data_x_node_type_transformed.append(1)
    if (item==edge_t2).all():
        data_x_node_type_transformed.append(2)

from torch_geometric.utils import k_hop_subgraph
from torch.nn.functional import cosine_similarity

# construct neighbouring graph
def get_1hop_complete_subgraph(data_edge_index, node_idx):
    """
    from the input graph, specify a node index, get 1-hop neighbourhood of the node,
    both from source to target (input node as target)
    and from target to source (input node as source)
    :param data: torch geometric data, full dataset
    :param node_idx: int, node index
    :return: tensor list of neighbouring node indices including node index, tensor list of edges
    """
    stt_subgraph_info = k_hop_subgraph(node_idx=node_idx, num_hops=1, edge_index=data_edge_index,
                                       relabel_nodes=False, flow="source_to_target")
    tts_subgraph_info = k_hop_subgraph(node_idx=node_idx, num_hops=1, edge_index=data_edge_index,
                                       relabel_nodes=False, flow="target_to_source")
    stt_nodes = stt_subgraph_info[0]
    tts_nodes = tts_subgraph_info[0]
    subg_nodes = torch.unique(torch.cat((stt_nodes, tts_nodes)))
    stt_edges = stt_subgraph_info[1].t()
    tts_edges = tts_subgraph_info[1].t()
    subg_edges = torch.unique(torch.cat((stt_edges, tts_edges), dim=0), dim=0)
    return subg_nodes, subg_edges

def graph_dissimilarity(data_x, data_x_node_type_transformed, edge_type_store, node_idx_1, subg_nodes_1, subg_edges_1, subg_nodes_2, subg_edges_2,
                        lamb_node=1, lamb_node_type=1, lamb_e=1, lamb_g=1):
    data_x_node_feat = data_x[:, :64]
    node_feat_dissim = node_features_dissimilarity(data_x_node_feat, node_idx_1, subg_nodes_2)
    node_type_dissim = node_type_dissimilarity(data_x_node_type_transformed, subg_nodes_1, subg_nodes_2)
    edge_feat_dissim = edge_features_dissimilarity(edge_type_store, subg_edges_1, subg_edges_2)
    graph_structure_dissim = graph_structure_dissimilarity(subg_edges_1, subg_edges_2)
    return lamb_node * node_feat_dissim + lamb_node_type*node_type_dissim + lamb_e * edge_feat_dissim + lamb_g * graph_structure_dissim, [node_feat_dissim, node_type_dissim, edge_feat_dissim, graph_structure_dissim]


# node features dissimilarity ranged [0, 1]:
# normalised L2 distance + cosine distance, between the input node 1 and the neighbouring nodes of node 2
def node_features_dissimilarity(data_x, node_idx_1, sug_nodes_2):
    feat1 = data_x[node_idx_1].view(1, -1)
    feat2 = data_x[sug_nodes_2]
    return (torch.norm(feat2 - feat1, p=2, dim=1).mean() / (
            (torch.norm(feat2, p=2, dim=1).mean()) + torch.norm(feat1, p=2)) + (
                    1 - (cosine_similarity(feat1, feat2).mean() + 1) / 2)) / 2

# Multiset Jaccard distance
def node_type_dissimilarity(data_x_node_type_transformed, subg_nodes_1, subg_nodes_2):
    # types_1 = np.array(data_x_node_type_transformed)[[subg_nodes_1]]
    # types_2 = np.array(data_x_node_type_transformed)[[subg_nodes_2]]
    # dissim_type = 0
    # for i in types_1.flatten():
    #     for j in types_2.flatten():
    #         dissim_type += int(i!=j)
    # return dissim_type / (len(types_1.flatten()) * len(types_2.flatten()))
    dict_a = get_node_type_count(data_x_node_type_transformed, subg_nodes_1)
    dict_b = get_node_type_count(data_x_node_type_transformed, subg_nodes_2)

    numerator = 0
    denominator = 0

    for i in [0, 1, 2]:
        try:
            count_a = dict_a[i]
        except:
            count_a = 0
        try:
            count_b = dict_b[i]
        except:
            count_b = 0
        numerator += min(count_a, count_b)
        denominator += max(count_a, count_b)
    return 1-numerator/denominator


def get_node_type_count(data_x_node_type_transformed, subg_nodes):
    a = np.array(data_x_node_type_transformed)[[subg_nodes]].flatten()
    unique_a = torch.unique(torch.from_numpy(a), return_counts=True)

    dict_a = {}
    for i, type in enumerate(unique_a[0]):
        dict_a[int(type)] = int(unique_a[1][i])
    return dict_a


    # edge features dissimilarity ranged [0, 1]:
def edge_features_dissimilarity(edge_idx_store, subg_edges_1, subg_edges_2):
    idxs1 = get_edge_type(edge_idx_store, subg_edges_1)
    idxs2 = get_edge_type(edge_idx_store, subg_edges_2)
    return (idxs1.mean() - idxs2.mean()).abs() / 4

def get_edge_type(edge_type_store, subg_edges):
    idxs = []
    for item in subg_edges:
        #print(f"{item[0]},{item[1]}")
        idxs.append(edge_type_store[f"{item[0]},{item[1]}"])
    return torch.Tensor(idxs).to(torch.float)

# graph structure dissimilarity
def graph_structure_dissimilarity(subg_edges_1, subg_edges_2):
    return abs(len(subg_edges_1) - len(subg_edges_2)) / max(len(subg_edges_1), len(subg_edges_2))

def compute_counterfactual(node_idx, target_prediction, predictions, data_edge_index, data_x, edge_type_store, data_x_node_type_transformed, mixed_idxs=None):
    """
    compute one counterfactual for node indicated by node_idx
    :param node_idx: int
    :param predictions: predictions, either regression or classification
    :param target_prediction
    :param data
    :return: counterfactual node, counterfactual subgraph, input subgraph, graph dissimilarity with input graph
    """
    if target_prediction == "mixed":
        candidate_ces_idxs = mixed_idxs
    else:
        candidate_ces_idxs = torch.where(predictions==target_prediction)[0]
    explainee_sgraph = get_1hop_complete_subgraph(data_edge_index, node_idx)
    optimal_ce_node = None
    optimal_ce_sgraph = None
    optimal_ce_dissimilarity = 100
    optimal_ce_dissimilarities = None
    for idx in candidate_ces_idxs:
        candidate_ce_sgraph = get_1hop_complete_subgraph(data_edge_index, int(idx))
        this_ce_dissimilarity, this_dissimilarities = graph_dissimilarity(data_x, data_x_node_type_transformed, edge_type_store, node_idx, explainee_sgraph[0], explainee_sgraph[1], candidate_ce_sgraph[0], candidate_ce_sgraph[1])
        if this_ce_dissimilarity <= optimal_ce_dissimilarity:
            optimal_ce_dissimilarity = this_ce_dissimilarity
            optimal_ce_node = idx
            optimal_ce_sgraph = candidate_ce_sgraph
            optimal_ce_dissimilarities = this_dissimilarities
    return optimal_ce_node, optimal_ce_sgraph, explainee_sgraph, optimal_ce_dissimilarity, optimal_ce_dissimilarities

# get all predictions and transform to classification labels
y_pred = model(data.x_dict, data.edge_index_dict)
y_pred_class = torch.argmax(y_pred, dim=1)

# construct a dict to store index for edge index
#edge_idx_store = {}
# for i, item in enumerate(data_edge_index.t()):
#     edge_idx_store[f"{item[0]},{item[1]}"] = i
edge_idx_store = edge_type_store

import numpy as np
import pandas as pd

def shannon_diversity_index(row):
    data = row[row > 0]
    probabilities = data / data.sum()
    return -np.sum(probabilities * np.log(probabilities))

# predicted labels df
label_columns_list = ['office', 'sustenance', 'transport', 'retail', 'leisure', 'residence']
y_pred = model(data.x_dict, data.edge_index_dict)
pred_df = pd.DataFrame(data=y_pred.detach().numpy(), columns=label_columns_list)
pred_df_shannon = pred_df[label_columns_list].apply(shannon_diversity_index, axis=1)
mixed_idxs = list(pred_df_shannon[pred_df_shannon>pred_df_shannon.quantile(0.8)].index)

target_prediction = "mixed"

# want to investigate the importance between node features, edge types, and graph structure, when changing from class 0 to mixed class.
input_idxs_with_non_mixed_class_0_label = list([int(i) for i in torch.where(y_pred_class==5)[0] if i not in mixed_idxs])
print(input_idxs_with_non_mixed_class_0_label)

dissimilarities = []
for idx in input_idxs_with_non_mixed_class_0_label:
    optimal_ce_node, optimal_ce_sgraph, explainee_sgraph, optimal_ce_dissimilarity, optimal_ce_dissimilarities = \
        compute_counterfactual(node_idx=idx, target_prediction=target_prediction, predictions=y_pred_class, data_edge_index=data_edge_index,
                               data_x=data_x, data_x_node_type_transformed=data_x_node_type_transformed, edge_type_store=edge_type_store, mixed_idxs=mixed_idxs)
    dissimilarities.append(optimal_ce_dissimilarities)
dissimilarities_tensor = torch.Tensor(dissimilarities)
# print(dissimilarities_tensor)

print(f"changing from class 0 to mixed neighbourhood, the average dissimilarities for node features, node type,  "
      f"edge type, and graph structure are respectively {np.round(float(dissimilarities_tensor.mean(axis=0)[0]),4)}+-"
      f"{np.round(float(dissimilarities_tensor.std(axis=0)[0]),6)}, {np.round(float(dissimilarities_tensor.mean(axis=0)[1]),4)}+-"
      f"{np.round(float(dissimilarities_tensor.std(axis=0)[1]),6)}, {np.round(float(dissimilarities_tensor.mean(axis=0)[2]),4)}+-"
      f"{np.round(float(dissimilarities_tensor.std(axis=0)[2]),6)}, {np.round(float(dissimilarities_tensor.mean(axis=0)[3]),4)}+-"
      f"{np.round(float(dissimilarities_tensor.std(axis=0)[3]),6)}, given that the three dissimilarity metrics are all in the same "
      f"scale ranged from 0-1, we can interpret the relative dissimilarity as relative importance")
