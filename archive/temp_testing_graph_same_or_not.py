import torch
from torch_geometric.data import Data

def remove_reserved_nodes_and_edges(data, reserved_node_mask):
    reserved_nodes = reserved_node_mask.nonzero().squeeze()
    data_sub = data.clone()
    edge_mask = torch.ones(data_sub.edge_index.size(1), dtype=torch.bool)
    for node in reserved_nodes:
        edge_mask[data_sub.edge_index[0] == node] = False
        edge_mask[data_sub.edge_index[1] == node] = False
    data_sub.edge_index = data_sub.edge_index[:, edge_mask]
    if hasattr(data_sub, 'edge_attr'):
        data_sub.edge_attr = data_sub.edge_attr[edge_mask]
    data_sub.x[reserved_nodes] = 0
    data_sub.train_mask[reserved_nodes] = False
    data_sub.val_mask[reserved_nodes] = False
    data_sub.test_mask[reserved_nodes] = False
    return data_sub

# 创建一个测试图
data = torch.load('data/processed/data.pt')

# 应用函数两次
result1 = remove_reserved_nodes_and_edges(data, data.test_mask)
result2 = remove_reserved_nodes_and_edges(data, data.test_mask)

# 比较结果
assert torch.all(result1.edge_index == result2.edge_index), "Edge indices are different"
assert torch.all(result1.x == result2.x), "Node features are different"
assert torch.all(result1.edge_attr == result2.edge_attr), "Edge weights are different"
assert torch.all(result1.train_mask == result2.train_mask), "Train masks are different"
assert torch.all(result1.val_mask == result2.val_mask), "Validation masks are different"
assert torch.all(result1.test_mask == result2.test_mask), "Test masks are different"

print("Test passed, function behaves consistently.")