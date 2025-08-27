import os, shutil
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import pickle
import argparse
from types import SimpleNamespace
from matplotlib import rcParams
rcParams.update({'figure.autolayout': False})
from tqdm import tqdm

from DataLoader import create_loaders, MeanTrainer, GIN, DiGCN, DiGCN_IB_Sum


root_path = r'C:/Users/yukun/Desktop/Test'
import warnings
warnings.filterwarnings("ignore")

#Step 1. 清除/processed/文件夹内的所有文件
folder = root_path + '/Data/BGL/processed'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

folder = root_path + '/Data/BGL/Raw'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

#Step 2. 将所有文件从一个目录复制到另一个目录
src_dir = root_path + '/Data/BGL/Graph/Raw/'
dest_dir = root_path + '/Data/BGL/Raw/'
my_files = os.listdir(src_dir)
for file_name in my_files:
    print(file_name)
    print(type(dest_dir))
    src_file_name = src_dir + file_name
    dest_file_name = dest_dir + file_name
    shutil.copy(src_file_name, dest_file_name)

#Step 3. 定义一个函数来运行实验
def run_experiment(
    data="BGL",                 # 数据集名称，默认为 BGL
    data_seed=1213,             # 随机种子，用于数据集划分，默认为 1213
    alpha=1.0,                  # SVDD 损失函数中的超参数 alpha，默认为 1.0
    beta=0.0,                   # SVDD 损失函数中的超参数 beta，默认为 0.0
    epochs=150,                 # 训练的 epoch 数，默认为 150
    model_seed=0,               # 模型的随机种子，默认为 0
    num_layers=1,               # 模型的隐藏层数，默认为 1
    device=0,                   # 使用的 GPU 设备编号，默认为 0
    aggregation="Mean",         # 图级别聚合方式，可选值为 {"Mean", "Max", "Sum"}，默认为 "Mean"
    bias=False,                 # 是否在 GNN 中使用偏置，默认为 False
    hidden_dim=64,              # 隐藏层维度，默认为 64
    lr=0.1,                     # 学习率，默认为 0.1
    weight_decay=1e-5,          # 权重衰减，默认为 1e-5
    batch=64                    # 批处理大小，默认为 64
):
    device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")

    # Step3.1. 使用预定义的脚本dataloader.py加载数据

    train_loader, test_loader, num_features, train_dataset, test_dataset, raw_dataset = create_loaders(data_name=data,batch_size=batch,dense=False,data_seed=data_seed)

    # 打印训练集和测试集数量及类别分布
    print("\nLoaded data details:")
    print("Number of training graphs: %d" % len(train_dataset))
    train_labels = np.array([data.y.item() for data in train_dataset])
    train_label_dist = ['%d' % (train_labels == c).sum() for c in [0, 1]]
    print("TRAIN: Class distribution %s" % train_label_dist)

    print("Number of testing graphs: %d" % len(test_dataset))
    test_labels = np.array([data.y.item() for data in test_dataset])
    test_label_dist = ['%d' % (test_labels == c).sum() for c in [0, 1]]
    print("TEST: Class distribution %s" % test_label_dist)
    #set seeds for cuda
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)


    #Step3.2. 用给定的参数训练GIN模型
    #设置参数
    model = DiGCN(nfeat=num_features, nhid=hidden_dim, nlayer=num_layers, bias=bias)

    #optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    if aggregation == "Mean":
        trainer = MeanTrainer(model=model,optimizer=optimizer,alpha=alpha,beta=beta,device=device)

    epochinfo = []      #用于存储每个epoch的训练和测试信息

    #开始训练
    for epoch in tqdm(range(epochs + 1)):
        print("Epoch %3d" % (epoch), end="\t")
        print("\n---------epoch train start-------------")
        svdd_loss = trainer.train(train_loader=train_loader)
        print("SVDD loss: %f" % (svdd_loss), end="\t")
        print("\n---------epoch train end-------------")
        print("\n---------epoch test start-------------")
        ap, roc_auc, dists, labels = trainer.test(test_loader=test_loader)
        print("ROC-AUC: %f" % roc_auc)
        print("\n---------epoch test end-------------")

        #设置临时对象以存储重要信息
        TEMP = SimpleNamespace()
        TEMP.epoch_no = epoch
        TEMP.dists = dists
        TEMP.labels = labels
        TEMP.ap = ap
        TEMP.roc_auc = roc_auc
        TEMP.svdd_loss = svdd_loss

        epochinfo.append(TEMP)

    best_svdd_idx = np.argmin([e.svdd_loss for e in epochinfo[1:]]) + 1

    print("Min SVDD, at epoch %d, AP: %.3f, ROC-AUC: %.3f" % (
    best_svdd_idx, epochinfo[best_svdd_idx].ap, epochinfo[best_svdd_idx].roc_auc))
    print("At the end, at epoch %d, AP: %.3f, ROC-AUC: %.3f" % (args.epochs, epochinfo[-1].ap, epochinfo[-1].roc_auc))

    ##----record the best epoch's information----

    important_epoch_info = {}
    important_epoch_info['svdd'] = epochinfo[best_svdd_idx]
    important_epoch_info['last'] = epochinfo[-1]

    # 保存最佳模型
    best_model = trainer.model
    torch.save(best_model.state_dict(), 'best_model.pth')
    return important_epoch_info, train_dataset, test_dataset, raw_dataset

# Step 4: 定义解析器
args = SimpleNamespace(
    data='BGL', batch=2000, data_seed=421, device=0, epochs=100, hidden_dim=300, layers=2, bias=False,
    aggregation='Mean', lr=0.01, weight_decay=1e-4, model_seed=0, use_config=True, config_file='configs/config.txt')

lrs = [args.lr]
weight_decays = [args.weight_decay]
layercounts = [args.layers]
model_seeds = [args.model_seed]

MyDict = {}

for lr in lrs:
    for weight_decay in weight_decays:
        for model_seed in model_seeds:
            for layercount in layercounts:
                print("Running experiment for LR=%f, weight decay = %.1E, model seed = %d, number of layers = %d" % (
                lr, weight_decay, model_seed, layercount))
                MyDict[(lr, weight_decay, model_seed, layercount)], my_train, my_test, my_raw_data = run_experiment(
                    data=args.data, data_seed=args.data_seed, epochs=args.epochs, model_seed=model_seed,
                    num_layers=layercount, device=args.device, aggregation=args.aggregation, bias=args.bias,
                    hidden_dim=args.hidden_dim, lr=lr, weight_decay=weight_decay, batch=args.batch
                )

if args.use_config:
    if not os.path.isdir('outputs'):
        os.mkdir('outputs')
    with open('outputs/GIN_' + args.aggregation + '_models_' + args.data + '_' + str(args.data_seed) + '.pkl',
              'wb') as f:
        pickle.dump(MyDict, f)

# 单个图形的可视化
test1 = my_raw_data[0]
import networkx as nx
import torch_geometric

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


