import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os  # 新增：用于创建文件夹


# 新增：创建.result文件夹（如果不存在）
os.makedirs(".result", exist_ok=True)

# 指定设备（GPU 或 CPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"当前使用设备：{device}")

# 数据准备
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 定义带有 Dropout 的神经网络
class MLPWithDropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(MLPWithDropout, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 初始化模型、损失函数和优化器
model = MLPWithDropout(dropout_rate=0.5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练函数（记录损失和准确率）
def train(model, train_loader, criterion, optimizer, epochs=5):
    train_losses = []
    train_accuracies = []
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 300 == 299:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss/300:.4f}')
                running_loss_batch = running_loss
                running_loss = 0.0
        
        epoch_loss = running_loss_batch / len(train_loader) * 300
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        print(f'Epoch [{epoch+1}/{epochs}] 训练准确率: {epoch_acc:.2f}%')
    
    return train_losses, train_accuracies

# 测试函数（收集预测结果和真实标签）
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    test_acc = 100 * correct / total
    print(f'Test Accuracy: {test_acc:.2f}%')
    return all_preds, all_targets, test_acc

# 修改：保存混淆矩阵为PDF（不显示图表）
def save_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('testlabel')
    plt.ylabel('truthlabel')
    plt.title('confusion matrix')
    plt.tight_layout()
    # 保存到.result文件夹
    plt.savefig("./result/confusion_matrix.pdf", format='pdf', bbox_inches='tight')
    plt.close()  # 关闭图表，释放内存

# 修改：保存训练曲线为PDF（不显示图表）
def save_training_curves(losses, accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    ax1.plot(range(1, len(losses)+1), losses, 'b-', marker='o')
    ax1.set_title('training loss curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('training loss')
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(range(1, len(accuracies)+1), accuracies, 'r-', marker='s')
    ax2.set_title('training accuracy curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('accuracy (%)')
    ax2.grid(True)
    ax2.set_ylim(80, 100)
    
    plt.tight_layout()
    # 保存到.result文件夹
    plt.savefig("./result/training_curves.pdf", format='pdf', bbox_inches='tight')
    plt.close()  # 关闭图表，释放内存

if __name__ == '__main__':
    train_losses, train_accuracies = train(model, train_loader, criterion, optimizer, epochs=5)
    all_preds, all_targets, test_acc = test(model, test_loader)
    class_names = [str(i) for i in range(10)]
    
    # 保存结果（替换原有的显示功能）
    save_confusion_matrix(all_targets, all_preds, class_names)
    save_training_curves(train_losses, train_accuracies)
    
    print("结果已保存到 result 文件夹，包含：")
    print("- confusion_matrix.pdf：混淆矩阵")
    print("- training_curves.pdf：训练损失和准确率曲线")
