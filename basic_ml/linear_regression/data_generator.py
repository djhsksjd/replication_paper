import numpy as np
import os

class LinearRegressionDataGenerator:
    """线性回归数据生成器"""
    
    def __init__(self, n_samples=1000, n_features=1, noise_level=0.1, random_state=42):
        """
        初始化数据生成器
        
        参数:
            n_samples: 样本数量
            n_features: 特征数量
            noise_level: 噪声水平
            random_state: 随机种子
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise_level = noise_level
        self.random_state = random_state
        
    def generate_data(self, coef=None, intercept=None):
        """
        生成线性回归数据
        
        参数:
            coef: 系数，如果为None则随机生成
            intercept: 截距，如果为None则随机生成
        
        返回:
            X: 特征数据 (n_samples, n_features)
            y: 目标值 (n_samples,)
            coef: 真实系数
            intercept: 真实截距
        """
        np.random.seed(self.random_state)
        
        # 生成特征数据
        X = np.random.randn(self.n_samples, self.n_features)
        
        # 如果未提供系数和截距，则随机生成
        if coef is None:
            coef = np.random.randn(self.n_features)
        if intercept is None:
            intercept = np.random.randn()
        
        # 生成目标值，添加噪声
        y = X.dot(coef) + intercept + np.random.randn(self.n_samples) * self.noise_level
        
        return X, y, coef, intercept
    
    def save_data(self, X, y, directory='data'):
        """
        保存数据到文件
        
        参数:
            X: 特征数据
            y: 目标值
            directory: 保存目录
        """
        # 创建目录（如果不存在）
        os.makedirs(directory, exist_ok=True)
        
        # 保存数据
        np.save(os.path.join(directory, 'X.npy'), X)
        np.save(os.path.join(directory, 'y.npy'), y)
        
        print(f"数据已保存到 {directory} 目录")

# 示例用法
if __name__ == "__main__":
    # 创建数据生成器
    generator = LinearRegressionDataGenerator(n_samples=1000, n_features=2, noise_level=0.5)
    
    # 生成数据
    X, y, coef, intercept = generator.generate_data()
    
    print(f"生成的数据形状: X={X.shape}, y={y.shape}")
    print(f"真实系数: {coef}")
    print(f"真实截距: {intercept}")
    
    # 保存数据
    generator.save_data(X, y)