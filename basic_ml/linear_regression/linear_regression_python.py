import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

class LinearRegressionPython:
    """使用Python实现的线性回归模型"""
    
    def __init__(self, learning_rate=0.01, n_iters=1000):
        """
        初始化线性回归模型
        
        参数:
            learning_rate: 学习率
            n_iters: 迭代次数
        """
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X, y):
        """
        训练线性回归模型
        
        参数:
            X: 特征数据 (n_samples, n_features)
            y: 目标值 (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # 初始化权重和偏置
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 梯度下降
        for i in range(self.n_iters):
            # 前向传播
            y_pred = self._predict(X)
            
            # 计算损失
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # 计算梯度
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # 每100次迭代打印一次损失
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{self.n_iters}, Loss: {loss:.4f}")
    
    def _predict(self, X):
        """
        使用当前参数进行预测
        
        参数:
            X: 特征数据
        
        返回:
            预测值
        """
        return np.dot(X, self.weights) + self.bias
    
    def predict(self, X):
        """
        预测新数据
        
        参数:
            X: 新的特征数据
        
        返回:
            预测值
        """
        return self._predict(X)
    
    def _compute_loss(self, y, y_pred):
        """
        计算均方误差损失
        
        参数:
            y: 真实值
            y_pred: 预测值
        
        返回:
            损失值
        """
        return np.mean((y_pred - y) ** 2)
    
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
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        return mse, r2
    
    def plot_loss(self):
        """
        绘制损失函数随迭代次数的变化
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.loss_history)), self.loss_history)
        plt.title('Loss vs. Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('loss_plot.png')
        plt.show()
        print("损失函数图像已保存为 loss_plot.png")
    
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
        plt.title('预测值与真实值对比')
        plt.xlabel(f'特征 {feature_idx+1}')
        plt.ylabel('目标值')
        plt.legend()
        plt.grid(True)
        plt.savefig('predictions_plot.png')
        plt.show()
        print("预测对比图像已保存为 predictions_plot.png")

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
        model = LinearRegressionPython(learning_rate=0.01, n_iters=1000)
        model.fit(X_train, y_train)
        
        # 评估模型
        mse, r2 = model.evaluate(X_test, y_test)
        print(f"测试集 MSE: {mse:.4f}")
        print(f"测试集 R²: {r2:.4f}")
        
        # 绘制损失函数
        model.plot_loss()
        
        # 如果是单特征数据，绘制预测对比图
        if X.shape[1] == 1:
            model.plot_predictions(X_test, y_test)
    else:
        print("数据文件不存在，请先运行data_generator.py生成数据")