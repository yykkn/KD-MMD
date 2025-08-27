import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from DataLoader2 import create_loaders, DiGCN
from Graph_Transformer_Networks2 import GraphTransformerNetworkWithAttention
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score

def calculate_metrics(model, loader):
    model.eval()
    y_true = []
    y_pred = []
    for data in loader:
        data = data.to(device)
        output = torch.sigmoid(model(data)).view(-1)
        pred = (output > 0.5).float()
        y_true.extend(data.y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    # 检查 y_true 是否包含两个类别
    if len(set(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_pred)
    else:
        roc_auc = None
    return precision, recall, f1, ap, roc_auc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_name = 'Hadoop'  # 替换为你的实际数据集名称
train_loader, test_loader, num_features, train_dataset, test_dataset, dataset_raw = create_loaders(data_name, batch_size=64)

model1 = DiGCN(nfeat=num_features, nhid=64, nlayer=2, dropout=0.5, out_dim=1).to(device)
optimizer1 = Adam(model1.parameters(), lr=0.001, weight_decay=5e-4)

model2 = GraphTransformerNetworkWithAttention(in_channels=num_features, hidden_channels=64, out_channels=1).to(device)
optimizer2 = Adam(model2.parameters(), lr=0.001, weight_decay=5e-4)

def train(model, optimizer, loader, other_model):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data).view(-1)
        loss = F.binary_cross_entropy_with_logits(output, data.y.float())
        other_output = other_model(data).view(-1)
        distillation_loss = F.kl_div(
            F.logsigmoid(output),
            F.sigmoid(other_output),
            reduction='batchmean'
        )
        loss += distillation_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    for data in loader:
        data = data.to(device)
        output = torch.sigmoid(model(data)).view(-1)
        pred = (output > 0.5).float()
        correct += pred.eq(data.y.float()).sum().item()
        total += len(data.y)
    return correct / total

best_acc1 = 0
best_acc2 = 0

for epoch in range(1, 101):
    loss1 = train(model1, optimizer1, train_loader, model2)
    loss2 = train(model2, optimizer2, train_loader, model1)
    train_acc1 = test(model1, train_loader)
    test_acc1 = test(model1, test_loader)
    train_acc2 = test(model2, train_loader)
    test_acc2 = test(model2, test_loader)
    print(f'Epoch: {epoch:03d}, Model1 Loss: {loss1:.4f}, Model1 Train Acc: {train_acc1:.4f}, Model1 Test Acc: {test_acc1:.4f}')
    print(f'Epoch: {epoch:03d}, Model2 Loss: {loss2:.4f}, Model2 Train Acc: {train_acc2:.4f}, Model2 Test Acc: {test_acc2:.4f}')

    if test_acc1 > best_acc1:
        best_acc1 = test_acc1
        torch.save(model1.state_dict(), 'model1_best_Hadoop.pth')
    if test_acc2 > best_acc2:
        best_acc2 = test_acc2
        torch.save(model2.state_dict(), 'model2_best_Hadoop.pth')

print("Calculating final metrics...")
precision1, recall1, f1_1, ap1, roc_auc1 = calculate_metrics(model1, test_loader)
precision2, recall2, f1_2, ap2, roc_auc2 = calculate_metrics(model2, test_loader)

if roc_auc1 is not None:
    print(f'Model1 Final Test Acc: {test_acc1:.4f}, Precision: {precision1:.4f}, Recall: {recall1:.4f}, F1: {f1_1:.4f}, AP: {ap1:.4f}, ROC AUC: {roc_auc1:.4f}')
else:
    print(f'Model1 Final Test Acc: {test_acc1:.4f}, Precision: {precision1:.4f}, Recall: {recall1:.4f}, F1: {f1_1:.4f}, AP: {ap1:.4f}, ROC AUC: Not Defined (only one class in y_true)')

if roc_auc2 is not None:
    print(f'Model2 Final Test Acc: {test_acc2:.4f}, Precision: {precision2:.4f}, Recall: {recall2:.4f}, F1: {f1_2:.4f}, AP: {ap2:.4f}, ROC AUC: {roc_auc2:.4f}')
else:
    print(f'Model2 Final Test Acc: {test_acc2:.4f}, Precision: {precision2:.4f}, Recall: {recall2:.4f}, F1: {f1_2:.4f}, AP: {ap2:.4f}, ROC AUC: Not Defined (only one class in y_true)')
