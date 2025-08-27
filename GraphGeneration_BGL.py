root_path = r'C:/Users/yukun/Desktop/Graph Neural Networks based Log Anomaly Detection and Explanation/Logs2Graph/'
import pandas as pd
from tqdm import tqdm
import pandas as pd
import numpy as np
import networkx as nx

MyDataName = "BGL"
df = pd.read_csv(root_path + '/Data/BGL/BGL.log_structured.csv', sep=',')
df['GroupId'] = df['Node']
raw_df = df[["LineId", "EventId", "GroupId", "EventTemplate"]]
import json
import pandas as pd

with open(root_path + '/Data/Gloves/Results/EmbeddingDict_BGL.json', 'r') as fp:
    embedding_dict = json.load(fp)
embedding_df = pd.DataFrame.from_dict(embedding_dict)


def GraphConstruction(my_example_df, graph_count_index, graph_loc_index, my_node_accum, new_node_accum):
    G = nx.MultiDiGraph()
    event_list = list(my_example_df["EventTemplate"])
    node_list = list(dict.fromkeys(event_list))
    G.add_nodes_from(node_list)
    G.add_edges_from([(event_list[v], event_list[v + 1]) for v in range(len(event_list) - 1)])
    A = nx.adjacency_matrix(G)
    df_A = pd.DataFrame(columns=["row", "column"])
    row_vec = (list(A.nonzero())[0]).tolist()
    col_vec = (list(A.nonzero())[1]).tolist()
    row_vec = [a + 1 for a in row_vec]
    col_vec = [a + 1 for a in col_vec]
    df_A["row"] = row_vec
    df_A["column"] = col_vec
    fp_A = root_path + "/Data/BGL/Graph/TempRaw/" + MyDataName + "_A.txt"
    np.savetxt(fp_A, df_A.values, fmt='%i', delimiter=', ')
    df_edge_weight = pd.DataFrame(columns=["edge_weight"])
    df_edge_weight["edge_weight"] = list(A.data)
    fp_edge_weight = root_path + "/Data/BGL/Graph/TempRaw/" + MyDataName + "_edge_attributes.txt"
    np.savetxt(fp_edge_weight, df_edge_weight.values, fmt='%i', delimiter=', ')
    df_graph_indicator = pd.DataFrame(columns=["indicator"])
    df_graph_indicator["indicator"] = [graph_count_index + 1] * len(new_node_accum)
    fp_graph_indicator = root_path + "/Data/BGL/Graph/TempRaw/" + MyDataName + "_graph_indicator.txt"
    np.savetxt(fp_graph_indicator, df_graph_indicator.values, fmt='%i', delimiter=', ')
    df_label = pd.read_csv(root_path + "/Data/BGL/anomaly_label.csv", sep=',')
    di_replace = {"Normal": 0, "Anomaly": 1}
    df_label = df_label.replace({"Label": di_replace})
    label_value = df_label.iloc[graph_loc_index]['Label']
    df_graph_labels = pd.DataFrame(columns=["labels"])
    df_graph_labels["labels"] = [label_value]
    fp_graph_labels = root_path + "/Data/BGL/Graph/TempRaw/" + MyDataName + "_graph_labels.txt"
    np.savetxt(fp_graph_labels, df_graph_labels.values, fmt='%i', delimiter=', ')
    node_attr_list = []
    mylist = event_list
    for node_name in list(dict.fromkeys(mylist)):
        arr_vec = embedding_df[node_name].values.tolist()
        node_attr_list.append(arr_vec)
    df_node_attributes = pd.DataFrame(node_attr_list)
    fp_node_attributes = root_path + "/Data/BGL/Graph/TempRaw/" + MyDataName + "_node_attributes.txt"
    np.savetxt(fp_node_attributes, df_node_attributes.values, fmt='%f', delimiter=', ')


import glob
import os
import os.path as osp
import torch
from torch_geometric.data import Data
from torch_geometric.io import read_txt_array
from torch_geometric.utils import coalesce, remove_self_loops
from torch_geometric.utils import to_undirected, is_undirected, to_networkx
from GetAdjacencyMatrix import get_undirected_adj, get_pr_directed_adj, get_appr_directed_adj, get_second_directed_adj, \
    get_appr_directed_adj_keep_attr


def read_file(folder, prefix, name, dtype=None):
    path = osp.join(folder, f'{prefix}_{name}.txt')
    return read_txt_array(path, sep=',', dtype=dtype)


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item for item in seq if item.numel() > 0]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1) if len(seq) > 0 else None


def read_tu_data(folder, prefix, adj_type):
    edge_index = read_file(folder, prefix, 'A', torch.long).t() - 1
    batch = read_file(folder, prefix, 'graph_indicator', torch.long) - 1
    if batch.dim() == 0:
        node_attributes = torch.empty((1, 0))
    else:
        node_attributes = torch.empty((batch.size(0), 0))
    node_attributes = read_file(folder, prefix, 'node_attributes')
    if node_attributes.dim() == 1:
        node_attributes = node_attributes.unsqueeze(-1)
    is_empty_index = 0
    if len(edge_index.shape) == 1:  ##some graph only have a single row
        is_empty_index = 1
        data = Data()
        print("---we have empty graph here---")
        return data, is_empty_index  ##if it is empty, we skip this dataset
        if edge_index.shape[0] == 0:
            edge_index = torch.tensor([[1], [1]])
            is_empty_index = 1
            data = Data()
            return data, is_empty_index
        else:
            edge_index = torch.tensor([[edge_index[0].item()], [edge_index[1].item()]])
    edge_attributes = torch.empty((edge_index.size(1), 0))
    edge_attributes = read_file(folder, prefix, 'edge_attributes')
    if edge_attributes.dim() == 1:
        edge_attributes = edge_attributes.unsqueeze(-1)
    x = cat([node_attributes])
    if edge_index.size(1) == 1:
        edge_attr = torch.tensor([[edge_attributes.item()]])
    else:
        edge_attr = cat([edge_attributes])
    y = None
    y = read_file(folder, prefix, 'graph_labels', torch.long)
    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)
    if edge_attr is None:
        edge_index = coalesce(edge_index, num_nodes=num_nodes)
    else:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes)
    if adj_type == 'un':
        print("\n Processing to undirected adj")
        indices = edge_index
        features = x
        indices = to_undirected(indices)
        edge_index, edge_attr = get_undirected_adj(edge_index=indices, num_nodes=features.shape[0],
                                                   dtype=features.dtype)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    elif adj_type == 'appr':
        alpha = 0.1
        indices = edge_index
        features = x
        edge_index, edge_attr = get_appr_directed_adj(alpha=alpha, edge_index=indices, num_nodes=features.shape[0],
                                                      dtype=features.dtype, edge_weight=edge_attr)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    elif adj_type == 'ib':
        alpha = 0.1
        indices = edge_index
        features = x
        edge_index, edge_attr = get_appr_directed_adj(alpha=alpha, edge_index=indices, num_nodes=features.shape[0],
                                                      dtype=features.dtype, edge_weight=edge_attr)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        edge_index2, edge_attr2 = get_second_directed_adj(edge_index=edge_index, num_nodes=features.shape[0],
                                                          dtype=features.dtype, edge_weight=edge_attr)

        data.edge_index2 = edge_index2
        data.edge_attr2 = edge_attr2
    return data, is_empty_index


def ConcatGraphs(ReadGraph, graph_count_index, my_node_accum, new_node_accum, adj_type):
    import pandas as pd
    import numpy as np
    fp_A = root_path + "/Data/BGL/Graph/Raw/" + MyDataName + "_A.txt"
    df_A = pd.DataFrame(ReadGraph.edge_index.numpy()).T
    df_A = df_A + my_node_accum
    with open(fp_A, "ab") as f:
        np.savetxt(f, df_A.values, fmt='%i', delimiter=', ')
    fp_edge_weight = root_path + "/Data/BGL/Graph/Raw/" + MyDataName + "_edge_attributes.txt"
    df_edge_weight = pd.DataFrame(ReadGraph.edge_attr.numpy())
    with open(fp_edge_weight, "ab") as f:
        np.savetxt(f, df_edge_weight.values, fmt='%f', delimiter=', ')
    df_graph_indicator = pd.DataFrame(columns=["indicator"])
    df_graph_indicator["indicator"] = [graph_count_index + 1] * len(new_node_accum)
    fp_graph_indicator = root_path + "/Data/BGL/Graph/Raw/" + MyDataName + "_graph_indicator.txt"
    with open(fp_graph_indicator, "ab") as f:
        np.savetxt(f, df_graph_indicator.values, fmt='%i', delimiter=', ')
    fp_graph_labels = root_path + "/Data/BGL/Graph/Raw/" + MyDataName + "_graph_labels.txt"
    df_graph_labels = pd.DataFrame([ReadGraph.y.numpy()])
    with open(fp_graph_labels, "ab") as f:
        np.savetxt(f, df_graph_labels.values, fmt='%i', delimiter=', ')
    fp_node_attributes = root_path + "/Data/BGL/Graph/Raw/" + MyDataName + "_node_attributes.txt"
    df_node_attributes = pd.DataFrame(ReadGraph.x.numpy())
    with open(fp_node_attributes, "ab") as f:
        np.savetxt(f, df_node_attributes.values, fmt='%f', delimiter=', ')
    if adj_type == 'ib':
        fp_A2 = root_path + "/Data/BGL/Graph/Raw/" + MyDataName + "_A2.txt"
        df_A2 = pd.DataFrame(ReadGraph.edge_index2.numpy()).T
        df_A2 = df_A2 + my_node_accum
        with open(fp_A2, "ab") as f:
            np.savetxt(f, df_A2.values, fmt='%i', delimiter=', ')
        fp_edge_weight2 = root_path + "/Data/BGL/Graph/Raw/" + MyDataName + "_edge_attributes2.txt"
        df_edge_weight2 = pd.DataFrame(ReadGraph.edge_attr2.numpy())
        with open(fp_edge_weight2, "ab") as f:
            np.savetxt(f, df_edge_weight2.values, fmt='%f', delimiter=', ')


import os, shutil

folder = root_path + "/Data/BGL/Graph/Raw"
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
all_event_df = pd.read_csv(root_path + '/Data/BGL/anomaly_label.csv', sep=',')
group_to_check = list(all_event_df["BlockId"])


def draw_sample(num_samples, anomaly_per, draw_code):
    if draw_code == 1:
        anomaly_index_list = all_event_df.index[all_event_df['Label'] == "Anomaly"].tolist()
        normal_index_list = all_event_df.index[all_event_df['Label'] == "Normal"].tolist()
        import random
        random.seed(50)
        anomal_drawn = random.sample(anomaly_index_list, int(num_samples * anomaly_per))
        normal_drawn = random.sample(normal_index_list, int(num_samples * (1 - anomaly_per)))
        sample_drawn = anomal_drawn + normal_drawn
        list_group = [group_to_check[my_idx] for my_idx in sample_drawn]
        return list_group, sample_drawn
    else:
        anomaly_index_list = all_event_df.index[all_event_df['Label'] == "Anomaly"].tolist()
        normal_index_list = all_event_df.index[all_event_df['Label'] == "Normal"].tolist()
        all_samples = anomaly_index_list + normal_index_list
        list_group = [group_to_check[my_idx] for my_idx in all_samples]
        return list_group, all_samples


num_graphs_to_test = 10000
anomaly_perentage = 0.45
draw_code_val = 1
list_group, list_group_idx = draw_sample(num_graphs_to_test, anomaly_perentage, draw_code_val)
all_event_list = []
count_index = 0
for group_name in tqdm(list_group):
    example_df = raw_df[raw_df["GroupId"] == group_name]
    node_accum = max(len(all_event_list) + 1, 1)
    new_event_list = list(dict.fromkeys(example_df["EventTemplate"]))
    GraphConstruction(my_example_df=example_df, graph_count_index=count_index,
                      graph_loc_index=list_group_idx[count_index], my_node_accum=node_accum,
                      new_node_accum=new_event_list)
    MyReadGraph, empty_index = read_tu_data(folder=root_path + "/Data/BGL/Graph/TempRaw", prefix=MyDataName,
                                            adj_type='ib')
    if empty_index == 0:
        ConcatGraphs(ReadGraph=MyReadGraph, graph_count_index=count_index, my_node_accum=node_accum,
                     new_node_accum=new_event_list, adj_type='ib')
        all_event_list += new_event_list
        count_index += 1
all_event_list = list(dict.fromkeys(all_event_list))  


