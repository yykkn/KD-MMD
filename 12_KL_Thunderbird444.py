import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score
from DataLoader5 import create_loaders, DiGCN
from Graph_Transformer_Networks2 import GraphTransformerNetworkWithAttention

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
    if len(set(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_pred)
    else:
        roc_auc = None
    return precision, recall, f1, ap, roc_auc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_name = 'Thunderbird'  # 替换为你的实际数据集名称
normal_train_loader, mixed_train_loader, test_loader, num_features, dataset_raw = create_loaders(data_name, batch_size=1)

model1 = DiGCN(nfeat=num_features, nhid=64, nlayer=2, dropout=0.5, out_dim=1).to(device)
optimizer1 = Adam(model1.parameters(), lr=0.001, weight_decay=5e-4)

model2 = GraphTransformerNetworkWithAttention(in_channels=num_features, hidden_channels=64, out_channels=1).to(device)
optimizer2 = Adam(model2.parameters(), lr=0.001, weight_decay=5e-4)

def train_one_sample(model1, optimizer1, model2, optimizer2, data, epoch):
    model1.train()
    model2.train()
    data = data.to(device)

    optimizer1.zero_grad()
    optimizer2.zero_grad()

    output1 = model1(data).view(-1)
    output2 = model2(data).view(-1)

    pred1 = (torch.sigmoid(output1) > 0.5).float()
    pred2 = (torch.sigmoid(output2) > 0.5).float()

    y = data.y.float()
    loss1 = F.binary_cross_entropy_with_logits(output1, y)
    loss2 = F.binary_cross_entropy_with_logits(output2, y)

    if epoch <= 10:
        if pred1.item() == y.item() and pred2.item() != y.item():
            distillation_loss = F.kl_div(
                F.logsigmoid(output2),
                F.sigmoid(output1).detach(),  # Detach to avoid in-place modification
                reduction='batchmean'
            )
            loss2 += distillation_loss

    else:
        if pred1.item() != y.item() and pred2.item() == y.item():
            distillation_loss = F.kl_div(
                F.logsigmoid(output1),
                F.sigmoid(output2).detach(),  # Detach to avoid in-place modification
                reduction='batchmean'
            )
            loss1 += distillation_loss

        elif pred1.item() == y.item() and pred2.item() != y.item():
            distillation_loss = F.kl_div(
                F.logsigmoid(output2),
                F.sigmoid(output1).detach(),  # Detach to avoid in-place modification
                reduction='batchmean'
            )
            loss2 += distillation_loss

    loss1.backward(retain_graph=True)
    optimizer1.step()

    loss2.backward()
    optimizer2.step()

    return loss1.item(), loss2.item()

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

for epoch in range(1, 11):
    if epoch <= 10:
        for data in normal_train_loader:
            train_one_sample(model1, optimizer1, model2, optimizer2, data, epoch)
    else:
        for data in mixed_train_loader:
            train_one_sample(model1, optimizer1, model2, optimizer2, data, epoch)

    train_acc1 = test(model1, normal_train_loader if epoch <= 10 else mixed_train_loader)
    train_acc2 = test(model2, normal_train_loader if epoch <= 10 else mixed_train_loader)
    test_acc1 = test(model1, test_loader)
    test_acc2 = test(model2, test_loader)

    print(f'Epoch {epoch}, Train Acc1: {train_acc1}, Test Acc1: {test_acc1}')
    print(f'Epoch {epoch}, Train Acc2: {train_acc2}, Test Acc2: {test_acc2}')

    if test_acc1 > best_acc1:
        best_acc1 = test_acc1
        torch.save(model1.state_dict(), 'model1_best_Thunderbird.pth')
    if test_acc2 > best_acc2:
        best_acc2 = test_acc2
        torch.save(model2.state_dict(), 'model2_best_Thunderbird.pth')

    # 保存最后一个epoch的模型
    if epoch == 100:
        torch.save(model1.state_dict(), 'model1_final_Thunderbird.pth')
        torch.save(model2.state_dict(), 'model2_final_Thunderbird.pth')
