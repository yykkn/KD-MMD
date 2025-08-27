import torch  # 导入PyTorch库
import torch.optim as optim  # 导入PyTorch优化器模块
from types import SimpleNamespace  # 导入SimpleNamespace类，用于创建简单的命名空间对象
from tqdm import tqdm  # 导入tqdm模块，用于在循环中显示进度条
import numpy as np  # 导入NumPy库，用于数学计算
from DataLoader import create_loaders, MeanTrainer, DiGCN  # 从DataLoader模块中导入create_loaders、MeanTrainer和DiGCN类

def run_experiment(data="BGL", data_seed=1213, alpha=1.0, beta=0.0, epochs=200, model_seed=0, num_layers=1,
                    device=0, aggregation="Mean", bias=False, hidden_dim=64, lr=0.002, weight_decay=1e-5, batch=64):
    # 根据CUDA设备是否可用选择设备
    device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")
    # 使用预定义的脚本dataloader.py加载数据
    train_loader, test_loader, num_features, train_dataset, test_dataset, raw_dataset = create_loaders(data_name=data,batch_size=batch,dense=False,data_seed=data_seed)
    # 设置参数
    model = DiGCN(nfeat=num_features, nhid=hidden_dim, nlayer=num_layers, bias=bias)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    if aggregation == "Mean":
        trainer = MeanTrainer(model=model, optimizer=optimizer, alpha=alpha, beta=beta, device=device)
    epochinfo = []  # 用于存储每个epoch的训练和测试信息

    # 开始训练
    for epoch in tqdm(range(epochs + 1)):
        svdd_loss = trainer.train(train_loader=train_loader)  # 训练模型
        ap, roc_auc, dists, labels = trainer.test(test_loader=test_loader)  # 测试模型
        TEMP = SimpleNamespace()
        TEMP.epoch_no = epoch
        TEMP.dists = dists
        TEMP.labels = labels
        TEMP.ap = ap
        TEMP.roc_auc = roc_auc
        TEMP.svdd_loss = svdd_loss
        epochinfo.append(TEMP)

    best_svdd_idx = np.argmin([e.svdd_loss for e in epochinfo[1:]]) + 1

    # 打印最小SVDD损失值的信息
    print("Min SVDD, at epoch %d, AP: %.3f, ROC-AUC: %.3f" % (best_svdd_idx, epochinfo[best_svdd_idx].ap, epochinfo[best_svdd_idx].roc_auc))
    # 打印最后一个epoch的信息
    print("At the end, at epoch %d, AP: %.3f, ROC-AUC: %.3f" % (epochs, epochinfo[-1].ap, epochinfo[-1].roc_auc))

    important_epoch_info = {}
    important_epoch_info['svdd'] = epochinfo[best_svdd_idx]
    important_epoch_info['last'] = epochinfo[-1]
    best_model = trainer.model
    torch.save(best_model.state_dict(), 'best_model.pth')  # 保存最佳模型参数到文件
    return important_epoch_info, train_dataset, test_dataset, raw_dataset  # 返回实验信息以及数据集

# 使用默认参数调用函数
run_experiment()
