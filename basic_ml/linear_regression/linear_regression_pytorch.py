import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, TensorDataset

class LinearRegressionPyTorch(nn.Module):
    """使用PyTorch实现的线性回归模型"""
    
    def __init__(self, input_dim):
        """
        初始化线性回归模型
        
        参数:
            input_dim: 输入特征维度
        """
        super(LinearRegressionPyTorch, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.loss_history = []
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征数据
        
        返回:
            预测值
        """
        return self.linear(x)
    
    def fit(self, X, y, learning_rate=0.01, n_iters=1000, batch_size=32):
        """
        训练线性回归模型
        
        参数:
            X: 特征数据 (n_samples, n_features)
            y: 目标值 (n_samples,)
            learning_rate: 学习率
            n_iters: 迭代次数
            batch_size: 批次大小
        """
        # 将数据转换为PyTorch张量
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        
        # 训练模型
        for epoch in range(n_iters):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                # 前向传播
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * batch_X.size(0)
            
            # 计算平均损失
            epoch_loss /= len(dataset)
            self.loss_history.append(epoch_loss)
            
            # 每100次迭代打印一次损失
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{n_iters}, Loss: {epoch_loss:.4f}")
    
    def predict(self, X):
        """
        预测新数据
        
        参数:
            X: 新的特征数据
        
        返回:
            预测值（numpy数组）
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
            
        # 设置为评估模式
        self.eval()
        
        with torch.no_grad():
            outputs = self(X)
        
        # 转回训练模式
        self.train()
        
        # 将结果转换为numpy数组
        return outputs.numpy().flatten()
    
    def evaluate(self, X, y):
        """
        评估模型性能
        
        参数:
            X: 特征数据
            y: 目标值
        
        返回:
            mse: 均方误差
            r2: R²分数
        """
        y_pred = self.predict(X)
        mse = np.mean((y_pred - y) ** 2)
        r2 = r2_score(y, y_pred)
        return mse, r2
    
    def plot_loss(self):
        """
        绘制损失函数随迭代次数的变化
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.loss_history)), self.loss_history)
        plt.title('Loss vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('loss_plot_pytorch.png')
        plt.show()
        print("损失函数图像已保存为 loss_plot_pytorch.png")
    
    def plot_predictions(self, X, y, feature_idx=0):
        """
        绘制预测值与真实值的对比图
        
        参数:
            X: 特征数据
            y: 目标值
            feature_idx: 要绘制的特征索引（仅适用于单特征情况）
        """
        if X.shape[1] > 1:
            print("警告：特征维度大于1，无法绘制二维散点图")
            return
        
        y_pred = self.predict(X)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(X[:, feature_idx], y, color='blue', label='真实值')
        plt.scatter(X[:, feature_idx], y_pred, color='red', label='预测值')
        plt.plot(X[:, feature_idx], y_pred, color='green', label='回归线')
        plt.title('预测值与真实值对比 (PyTorch)')
        plt.xlabel(f'特征 {feature_idx+1}')
        plt.ylabel('目标值')
        plt.legend()
        plt.grid(True)
        plt.savefig('predictions_plot_pytorch.png')
        plt.show()
        print("预测对比图像已保存为 predictions_plot_pytorch.png")
    
    def save_model(self, path='linear_regression_model.pth'):
        """
        保存模型
        
        参数:
            path: 模型保存路径
        """
        torch.save(self.state_dict(), path)
        print(f"模型已保存到 {path}")
    
    def load_model(self, path='linear_regression_model.pth'):
        """
        加载模型
        
        参数:
            path: 模型加载路径
        """
        self.load_state_dict(torch.load(path))
        print(f"模型已从 {path} 加载")

# 示例用法
if __name__ == "__main__":
    # 检查是否存在数据文件
    if os.path.exists('data/X.npy') and os.path.exists('data/y.npy'):
        # 加载数据
        X = np.load('data/X.npy')
        y = np.load('data/y.npy')
        
        # 划分训练集和测试集
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 创建并训练模型
        input_dim = X.shape[1]
        model = LinearRegressionPyTorch(input_dim)
        model.fit(X_train, y_train, learning_rate=0.01, n_iters=1000, batch_size=32)
        
        # 评估模型
        mse, r2 = model.evaluate(X_test, y_test)
        print(f"测试集 MSE: {mse:.4f}")
        print(f"测试集 R²: {r2:.4f}")
        
        # 绘制损失函数
        model.plot_loss()
        
        # 如果是单特征数据，绘制预测对比图
        if X.shape[1] == 1:
            model.plot_predictions(X_test, y_test)
            
        # 保存模型
        model.save_model()
    else:
        print("数据文件不存在，请先运行data_generator.py生成数据")