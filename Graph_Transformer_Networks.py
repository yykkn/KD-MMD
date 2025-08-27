# Graph_Transformer_Networks.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GraphTransformerNetwork(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GraphTransformerNetwork, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # 全局均值池化，将每个图的节点特征聚合成图级别特征
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        x = torch.sigmoid(x)  # 添加 Sigmoid 激活函数
        return x
