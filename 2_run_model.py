import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from Graph_Transformer_Networks import GraphTransformerNetwork
from DataLoader import create_loaders

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
data_name = 'BGL'  # 替换为你的实际数据集名称
train_loader, test_loader, num_features, train_dataset, test_dataset, dataset_raw = create_loaders(data_name, batch_size=64)

# 初始化模型
model = GraphTransformerNetwork(in_channels=num_features, hidden_channels=64, out_channels=2, dropout=0.5).to(device)
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# 定义训练和测试函数
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

# 训练和测试循环
for epoch in range(1, 201):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
