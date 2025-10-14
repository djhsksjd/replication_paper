import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import KFold

class LinearRegressionEnhanced:
    """增强版线性回归模型，支持L1/L2正则化、交叉验证和多种优化算法"""
    
    def __init__(self, learning_rate=0.01, n_iters=1000, penalty='none', 
                 l1_strength=0.01, l2_strength=0.01, optimizer='sgd', 
                 momentum=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        初始化增强版线性回归模型
        
        参数:
            learning_rate: 学习率
            n_iters: 迭代次数
            penalty: 正则化类型 ('none', 'l1', 'l2', 'elasticnet')
            l1_strength: L1正则化强度
            l2_strength: L2正则化强度
            optimizer: 优化算法 ('sgd', 'momentum', 'adam', 'rmsprop')
            momentum: 动量参数（用于momentum优化器）
            beta1: Adam优化器的一阶矩估计衰减率
            beta2: Adam优化器的二阶矩估计衰减率
            epsilon: 数值稳定性参数
        """
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.penalty = penalty.lower()
        self.l1_strength = l1_strength
        self.l2_strength = l2_strength
        self.optimizer = optimizer.lower()
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.weights = None
        self.bias = None
        self.loss_history = []
        
        # 检查参数有效性
        if self.penalty not in ['none', 'l1', 'l2', 'elasticnet']:
            raise ValueError("正则化类型必须是'none', 'l1', 'l2'或'elasticnet'")
        
        if self.optimizer not in ['sgd', 'momentum', 'adam', 'rmsprop']:
            raise ValueError("优化器必须是'sgd', 'momentum', 'adam'或'rmsprop'")
    
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
        
        # 初始化优化器状态
        if self.optimizer == 'momentum':
            v_w = np.zeros(n_features)
            v_b = 0
        elif self.optimizer == 'adam':
            m_w = np.zeros(n_features)
            v_w = np.zeros(n_features)
            m_b = 0
            v_b = 0
            t = 0
        elif self.optimizer == 'rmsprop':
            cache_w = np.zeros(n_features)
            cache_b = 0
        
        # 梯度下降
        for i in range(self.n_iters):
            # 前向传播
            y_pred = self._predict(X)
            
            # 计算损失（包括正则化项）
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # 计算梯度
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # 添加正则化梯度
            if self.penalty == 'l1' or self.penalty == 'elasticnet':
                dw += self.l1_strength * np.sign(self.weights)
            if self.penalty == 'l2' or self.penalty == 'elasticnet':
                dw += 2 * self.l2_strength * self.weights
            
            # 根据优化算法更新参数
            if self.optimizer == 'sgd':
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            elif self.optimizer == 'momentum':
                v_w = self.momentum * v_w + self.learning_rate * dw
                v_b = self.momentum * v_b + self.learning_rate * db
                self.weights -= v_w
                self.bias -= v_b
            elif self.optimizer == 'adam':
                t += 1
                m_w = self.beta1 * m_w + (1 - self.beta1) * dw
                v_w = self.beta2 * v_w + (1 - self.beta2) * (dw ** 2)
                m_b = self.beta1 * m_b + (1 - self.beta1) * db
                v_b = self.beta2 * v_b + (1 - self.beta2) * (db ** 2)
                
                # 偏差校正
                m_w_hat = m_w / (1 - self.beta1 ** t)
                v_w_hat = v_w / (1 - self.beta2 ** t)
                m_b_hat = m_b / (1 - self.beta1 ** t)
                v_b_hat = v_b / (1 - self.beta2 ** t)
                
                self.weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
                self.bias -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
            elif self.optimizer == 'rmsprop':
                cache_w = self.beta1 * cache_w + (1 - self.beta1) * (dw ** 2)
                cache_b = self.beta1 * cache_b + (1 - self.beta1) * (db ** 2)
                
                self.weights -= self.learning_rate * dw / (np.sqrt(cache_w) + self.epsilon)
                self.bias -= self.learning_rate * db / (np.sqrt(cache_b) + self.epsilon)
            
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
        计算损失函数（包括正则化项）
        
        参数:
            y: 真实值
            y_pred: 预测值
        
        返回:
            损失值
        """
        # 均方误差
        mse = np.mean((y_pred - y) ** 2)
        
        # 添加正则化项
        if self.penalty == 'l1':
            mse += self.l1_strength * np.sum(np.abs(self.weights))
        elif self.penalty == 'l2':
            mse += self.l2_strength * np.sum(self.weights ** 2)
        elif self.penalty == 'elasticnet':
            mse += self.l1_strength * np.sum(np.abs(self.weights)) + \
                   self.l2_strength * np.sum(self.weights ** 2)
        
        return mse
    
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
    
    def cross_validate(self, X, y, n_splits=5):
        """
        执行k折交叉验证
        
        参数:
            X: 特征数据
            y: 目标值
            n_splits: 交叉验证的折数
        
        返回:
            mse_scores: 各折的均方误差
            r2_scores: 各折的R²分数
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        mse_scores = []
        r2_scores = []
        
        print(f"执行{kf.get_n_splits(X)}折交叉验证...")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 创建新模型实例进行训练
            fold_model = LinearRegressionEnhanced(
                learning_rate=self.learning_rate,
                n_iters=self.n_iters,
                penalty=self.penalty,
                l1_strength=self.l1_strength,
                l2_strength=self.l2_strength,
                optimizer=self.optimizer,
                momentum=self.momentum,
                beta1=self.beta1,
                beta2=self.beta2,
                epsilon=self.epsilon
            )
            
            # 训练模型
            fold_model.fit(X_train, y_train)
            
            # 评估模型
            mse, r2 = fold_model.evaluate(X_val, y_val)
            mse_scores.append(mse)
            r2_scores.append(r2)
            
            print(f"折 {fold+1}: MSE = {mse:.4f}, R² = {r2:.4f}")
        
        print(f"交叉验证结果 - 平均MSE: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
        print(f"交叉验证结果 - 平均R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
        
        return mse_scores, r2_scores
    
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
        plt.savefig('loss_plot_enhanced.png')
        plt.show()
        print("损失函数图像已保存为 loss_plot_enhanced.png")
    
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
        plt.title('预测值与真实值对比 (增强版)')
        plt.xlabel(f'特征 {feature_idx+1}')
        plt.ylabel('目标值')
        plt.legend()
        plt.grid(True)
        plt.savefig('predictions_plot_enhanced.png')
        plt.show()
        print("预测对比图像已保存为 predictions_plot_enhanced.png")
    
    def compare_optimizers(self, X, y):
        """
        比较不同优化器的性能
        
        参数:
            X: 特征数据
            y: 目标值
        """
        optimizers = ['sgd', 'momentum', 'adam', 'rmsprop']
        histories = {}
        
        # 划分训练集和验证集
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("比较不同优化器的性能...")
        
        # 使用不同的优化器训练模型
        for opt in optimizers:
            print(f"\n使用{opt.upper()}优化器:")
            model = LinearRegressionEnhanced(
                learning_rate=self.learning_rate,
                n_iters=self.n_iters,
                penalty=self.penalty,
                l1_strength=self.l1_strength,
                l2_strength=self.l2_strength,
                optimizer=opt
            )
            model.fit(X_train, y_train)
            histories[opt] = model.loss_history
            
            # 评估模型
            mse, r2 = model.evaluate(X_val, y_val)
            print(f"验证集 MSE: {mse:.4f}")
            print(f"验证集 R²: {r2:.4f}")
        
        # 绘制不同优化器的损失曲线对比
        plt.figure(figsize=(12, 8))
        for opt, history in histories.items():
            plt.plot(range(len(history)), history, label=opt.upper())
        
        plt.title('不同优化器的损失函数对比')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('optimizers_comparison.png')
        plt.show()
        print("优化器对比图像已保存为 optimizers_comparison.png")
        
        return histories
    
    def hyperparameter_tuning(self, X, y, param_grid, n_splits=5):
        """
        超参数调优
        
        参数:
            X: 特征数据
            y: 目标值
            param_grid: 超参数网格
            n_splits: 交叉验证的折数
        
        返回:
            best_params: 最佳超参数组合
            best_score: 最佳评分
        """
        from itertools import product
        import copy
        
        # 生成超参数组合
        param_combinations = list(product(*param_grid.values()))
        param_names = list(param_grid.keys())
        
        best_score = -float('inf')  # 对于R²，我们希望最大化分数
        best_params = None
        
        print(f"超参数调优: 共有{len(param_combinations)}种参数组合")
        
        # 遍历所有参数组合
        for i, params in enumerate(param_combinations):
            param_dict = dict(zip(param_names, params))
            
            print(f"\n测试参数组合 {i+1}/{len(param_combinations)}: {param_dict}")
            
            # 创建模型
            model = LinearRegressionEnhanced(
                learning_rate=self.learning_rate,
                n_iters=self.n_iters,
                penalty=self.penalty,
                l1_strength=self.l1_strength,
                l2_strength=self.l2_strength,
                optimizer=self.optimizer,
                **param_dict  # 覆盖默认参数
            )
            
            # 执行交叉验证
            _, r2_scores = model.cross_validate(X, y, n_splits)
            avg_r2 = np.mean(r2_scores)
            
            # 更新最佳参数
            if avg_r2 > best_score:
                best_score = avg_r2
                best_params = copy.deepcopy(param_dict)
                print(f"找到更好的参数组合，平均R²: {avg_r2:.4f}")
        
        print(f"\n超参数调优完成!")
        print(f"最佳参数: {best_params}")
        print(f"最佳平均R²: {best_score:.4f}")
        
        return best_params, best_score

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
        
        # 创建并训练模型（带L2正则化和Adam优化器）
        model = LinearRegressionEnhanced(
            learning_rate=0.01,
            n_iters=1000,
            penalty='l2',
            l2_strength=0.01,
            optimizer='adam'
        )
        model.fit(X_train, y_train)
        
        # 评估模型
        mse, r2 = model.evaluate(X_test, y_test)
        print(f"\n测试集 MSE: {mse:.4f}")
        print(f"测试集 R²: {r2:.4f}")
        
        # 执行交叉验证
        model.cross_validate(X_train, y_train, n_splits=5)
        
        # 绘制损失函数
        model.plot_loss()
        
        # 如果是单特征数据，绘制预测对比图
        if X.shape[1] == 1:
            model.plot_predictions(X_test, y_test)
            
        # 比较不同优化器（可选）
        # model.compare_optimizers(X, y)
        
        # 超参数调优（可选）
        # param_grid = {
        #     'learning_rate': [0.001, 0.01, 0.1],
        #     'l2_strength': [0.001, 0.01, 0.1]
        # }
        # best_params, best_score = model.hyperparameter_tuning(X_train, y_train, param_grid)
    else:
        print("数据文件不存在，请先运行data_generator.py生成数据")