import os
import matplotlib.pyplot as plt
import networkx as nx
import torch_geometric
import pickle

# 单个图形的可视化
test1 = my_raw_data[0]
# 边索引和边属性
edge_index = test1.edge_index.numpy()
edge_attr = test1.edge_attr.numpy()
# 获取边的权重（频次）
edge_weights = {(edge_index[0, i], edge_index[1, i]): edge_attr[i, 0] for i in range(edge_index.shape[1])}

g = torch_geometric.utils.to_networkx(test1, to_undirected=False)
# 获取边的权重（频次）
edge_labels = nx.get_edge_attributes(g, 'weight')

plt.figure(figsize=(10, 8))  # 设置图形大小
pos = nx.spring_layout(g)  # 使用spring布局进行绘制
nx.draw(g, pos, with_labels=True, node_color='lightblue')
# 在边上显示权重
nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_weights)
plt.savefig(os.path.join('outputs', 'graph_visualization.png'))  # 保存图形
plt.close()  # 关闭图形窗口
