import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from DataLoader2 import create_loaders, DiGCN
from Graph_Transformer_Networks2 import GraphTransformerNetworkWithAttention
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score


def calculate_metrics(model, loader):
    # 设置模型为评估模式
    model.eval()
    y_true = []  # 用于存储真实标签的列表
    y_pred = []  # 用于存储预测标签的列表

    # 遍历数据加载器中的每个批次
    for data in loader:
        data = data.to(device)  # 将数据移动到指定设备（如GPU）
        output = torch.sigmoid(model(data)).view(-1)  # 计算模型输出，并应用sigmoid函数，将输出展平为一维
        pred = (output > 0.5).float()  # 预测标签，大于0.5的为正例（1），否则为负例（0）

        # 将真实标签和预测标签从GPU移到CPU，并转换为NumPy数组，然后扩展到列表中
        y_true.extend(data.y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

    # 使用sklearn计算精确度、召回率、F1分数、平均精确度
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)

    # 计算ROC AUC分数，如果y_true包含两个类别
    if len(set(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_pred)
    else:
        roc_auc = None

    return precision, recall, f1, ap, roc_auc


# 加载设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_name = 'Thunderbird'  # 替换为你的实际数据集名称

# 创建数据加载器
train_loader, test_loader, num_features, train_dataset, test_dataset, dataset_raw = create_loaders(data_name,
                                                                                                   batch_size=64)

# 定义模型1和模型2的结构
model1 = DiGCN(nfeat=num_features, nhid=64, nlayer=2, dropout=0.5, out_dim=1).to(device)
model2 = GraphTransformerNetworkWithAttention(in_channels=num_features, hidden_channels=64, out_channels=1).to(device)

# 加载保存的模型参数
model1.load_state_dict(torch.load('model1_final_Thunderbird.pth'))
model2.load_state_dict(torch.load('model2_final_Thunderbird.pth'))

# 计算模型1在测试集上的指标
precision1, recall1, f1_1, ap1, roc_auc1 = calculate_metrics(model1, test_loader)

# 输出模型1的指标
if roc_auc1 is not None:
    print(f'Model1 Test Acc: {f1_1:.4f}, Precision: {precision1:.4f}, Recall: {recall1:.4f}, F1: {f1_1:.4f}, AP: {ap1:.4f}, ROC AUC: {roc_auc1:.4f}')
else:
    print(f'Model1 Test Acc: {f1_1:.4f}, Precision: {precision1:.4f}, Recall: {recall1:.4f}, F1: {f1_1:.4f}, AP: {ap1:.4f}, ROC AUC: Not Defined (only one class in y_true)')

# 计算模型2在测试集上的指标
precision2, recall2, f1_2, ap2, roc_auc2 = calculate_metrics(model2, test_loader)

# 输出模型2的指标
if roc_auc2 is not None:
    print(f'Model2 Test Acc: {f1_2:.4f}, Precision: {precision2:.4f}, Recall: {recall2:.4f}, F1: {f1_2:.4f}, AP: {ap2:.4f}, ROC AUC: {roc_auc2:.4f}')
else:
    print(f'Model2 Test Acc: {f1_2:.4f}, Precision: {precision2:.4f}, Recall: {recall2:.4f}, F1: {f1_2:.4f}, AP: {ap2:.4f}, ROC AUC: Not Defined (only one class in y_true)')
