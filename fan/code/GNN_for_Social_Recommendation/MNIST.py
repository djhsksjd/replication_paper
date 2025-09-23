import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. 数据准备：加载MNIST数据集并进行预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化（MNIST数据集的均值和标准差）
])

# 下载并加载训练集和测试集
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 2. 定义简单的神经网络（多层感知机）
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # 输入层到隐藏层1：输入维度784（28x28像素），输出维度256
        self.fc1 = nn.Linear(784, 256)
        # 隐藏层1到隐藏层2：输入维度256，输出维度128
        self.fc2 = nn.Linear(256, 128)
        # 隐藏层2到输出层：输入维度128，输出维度10（10个数字类别）
        self.fc3 = nn.Linear(128, 10)
        # 激活函数：ReLU
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 将输入的28x28图像展平为784维向量
        x = x.view(-1, 784)
        # 输入层 -> 隐藏层1，应用ReLU激活
        x = self.relu(self.fc1(x))
        # 隐藏层1 -> 隐藏层2，应用ReLU激活
        x = self.relu(self.fc2(x))
        # 隐藏层2 -> 输出层，不应用激活（后续使用CrossEntropyLoss包含softmax）
        x = self.fc3(x)
        return x

# 3. 初始化模型、损失函数和优化器
model = SimpleMLP()
criterion = nn.CrossEntropyLoss()  # 交叉熵损失，适用于分类任务
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # 随机梯度下降优化器

# 4. 训练模型
def train(model, train_loader, criterion, optimizer, epochs=5):
    model.train()  # 切换到训练模式
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            # 清零梯度
            optimizer.zero_grad()
            # 前向传播：计算模型输出
            output = model(data)
            # 计算损失
            loss = criterion(output, target)
            # 反向传播：计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()
            
            # 打印训练进度
            running_loss += loss.item()
            if batch_idx % 300 == 299:  # 每300个批次打印一次
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss/300:.4f}')
                running_loss = 0.0

# 5. 测试模型
def test(model, test_loader):
    model.eval()  # 切换到评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 关闭梯度计算，节省内存
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)  # 获取预测类别
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print(f'Test Accuracy: {100 * correct / total:.2f}%')

# 6. 执行训练和测试
if __name__ == '__main__':
    train(model, train_loader, criterion, optimizer, epochs=5)
    test(model, test_loader)
