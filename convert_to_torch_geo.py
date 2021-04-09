"""
Azad Amitoz 2021-04-09 16:00
"""
import torch
from torch_geometric.data import Data
import numpy as np

# load the pointcloud
x = np.load("./data/X.npz")["position"]
print(x.shape)

# load the graph
graph = np.load("./graphs/graph_cpu.npz")
edge_index = graph["ngbrs"] 
k = graph["k"]
wgts = graph["weights"]

# convert to torch_geo format
index = np.zeros(len(edge_index))
for u in range(len(x)):
    index[u*k:(u + 1) * k] = np.kron(np.ones((k)), u)
edge_index = np.stack([edge_index, index],0)
edge_index = torch.tensor(edge_index, dtype=torch.long)
edge_attr = torch.tensor(wgts, dtype=torch.float32)
edge_attr = edge_attr.view(-1,1)
graph = Data(x=x , edge_index = edge_index, edge_attr=edge_attr)
torch.save(graph,"/tmp/graph.pt")
