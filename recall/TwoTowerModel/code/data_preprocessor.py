import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import pickle

class DataPreprocessor:
    def __init__(self, data_dir):
        """
        初始化数据预处理器
        :param data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.user_features = None
        self.item_features = None
        
    def load_data(self, interactions_file='interactions.csv', user_features_file=None, item_features_file=None):
        """
        加载交互数据和特征数据
        :param interactions_file: 交互数据文件名
        :param user_features_file: 用户特征数据文件名
        :param item_features_file: 物品特征数据文件名
        :return: 处理后的交互数据
        """
        # 加载交互数据
        interactions_path = os.path.join(self.data_dir, interactions_file)
        interactions = pd.read_csv(interactions_path)
        
        # 确保必要的列存在
        required_cols = ['user_id', 'item_id', 'rating']
        for col in required_cols:
            if col not in interactions.columns:
                raise ValueError(f"交互数据缺少必要的列: {col}")
        
        # 编码用户ID和物品ID
        interactions['user_id_encoded'] = self.user_encoder.fit_transform(interactions['user_id'])
        interactions['item_id_encoded'] = self.item_encoder.fit_transform(interactions['item_id'])
        
        # 加载用户特征（如果提供）
        if user_features_file:
            user_features_path = os.path.join(self.data_dir, user_features_file)
            self.user_features = pd.read_csv(user_features_path)
            self.user_features['user_id_encoded'] = self.user_encoder.transform(self.user_features['user_id'])
        
        # 加载物品特征（如果提供）
        if item_features_file:
            item_features_path = os.path.join(self.data_dir, item_features_file)
            self.item_features = pd.read_csv(item_features_path)
            self.item_features['item_id_encoded'] = self.item_encoder.transform(self.item_features['item_id'])
        
        return interactions
    
    def split_data(self, interactions, test_size=0.2, val_size=0.1):
        """
        分割数据集为训练集、验证集和测试集
        :param interactions: 交互数据
        :param test_size: 测试集比例
        :param val_size: 验证集比例
        :return: 训练集、验证集、测试集
        """
        # 按用户分割数据，确保每个用户在训练集、验证集和测试集中都有数据
        user_groups = interactions.groupby('user_id_encoded')
        train_data = []
        val_data = []
        test_data = []
        
        for _, group in user_groups:
            # 按时间排序（假设数据中有序号列）
            if 'timestamp' in group.columns:
                group = group.sort_values('timestamp')
            else:
                group = group.reset_index(drop=True)
            
            # 分割数据
            n = len(group)
            test_idx = int(n * (1 - test_size))
            val_idx = int(test_idx * (1 - val_size))
            
            train_data.append(group.iloc[:val_idx])
            val_data.append(group.iloc[val_idx:test_idx])
            test_data.append(group.iloc[test_idx:])
        
        # 合并数据
        train_data = pd.concat(train_data).reset_index(drop=True)
        val_data = pd.concat(val_data).reset_index(drop=True)
        test_data = pd.concat(test_data).reset_index(drop=True)
        
        return train_data, val_data, test_data
    
    def create_negatives(self, interactions, num_negatives=4):
        """
        为每个正样本创建负样本
        :param interactions: 交互数据
        :param num_negatives: 每个正样本对应的负样本数量
        :return: 包含正负样本的数据
        """
        # 获取所有用户和物品的唯一ID
        all_users = interactions['user_id_encoded'].unique()
        all_items = interactions['item_id_encoded'].unique()
        
        # 创建用户-物品交互的集合，用于快速查找
        user_item_set = set(zip(interactions['user_id_encoded'], interactions['item_id_encoded']))
        
        # 创建负样本
        negatives = []
        for user_id in all_users:
            # 找到该用户已交互的物品
            interacted_items = set(interactions[interactions['user_id_encoded'] == user_id]['item_id_encoded'])
            # 找到该用户未交互的物品
            non_interacted_items = list(set(all_items) - interacted_items)
            
            # 为每个正样本随机选择负样本
            user_interactions = interactions[interactions['user_id_encoded'] == user_id]
            for _, row in user_interactions.iterrows():
                # 添加正样本
                positives = [(row['user_id_encoded'], row['item_id_encoded'], 1)]
                # 随机选择负样本
                neg_items = np.random.choice(non_interacted_items, min(num_negatives, len(non_interacted_items)), replace=False)
                neg_samples = [(user_id, item_id, 0) for item_id in neg_items]
                # 合并正负样本
                negatives.extend(positives + neg_samples)
        
        # 转换为DataFrame
        neg_df = pd.DataFrame(negatives, columns=['user_id', 'item_id', 'label'])
        return neg_df
    
    def save_processed_data(self, train_data, val_data, test_data, output_dir):
        """
        保存处理后的数据
        :param train_data: 训练数据
        :param val_data: 验证数据
        :param test_data: 测试数据
        :param output_dir: 输出目录
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存数据
        train_data.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
        val_data.to_csv(os.path.join(output_dir, 'val_data.csv'), index=False)
        test_data.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)
        
        # 保存编码器和特征信息
        with open(os.path.join(output_dir, 'encoders.pkl'), 'wb') as f:
            pickle.dump({
                'user_encoder': self.user_encoder,
                'item_encoder': self.item_encoder
            }, f)
    
    def get_feature_dimensions(self):
        """
        获取特征维度信息
        :return: 特征维度字典
        """
        return {
            'num_users': len(self.user_encoder.classes_),
            'num_items': len(self.item_encoder.classes_)
        }

# 示例用法
if __name__ == '__main__':
    # 创建示例数据
    import pandas as pd
    import numpy as np
    
    # 创建示例交互数据
    np.random.seed(42)
    num_users = 100
    num_items = 50
    num_interactions = 1000
    
    user_ids = np.random.randint(0, num_users, num_interactions)
    item_ids = np.random.randint(0, num_items, num_interactions)
    ratings = np.random.randint(1, 6, num_interactions)
    
    # 创建唯一的用户-物品对
    unique_pairs = []
    seen = set()
    for u, i, r in zip(user_ids, item_ids, ratings):
        if (u, i) not in seen:
            unique_pairs.append((u, i, r))
            seen.add((u, i))
    
    interactions = pd.DataFrame(unique_pairs, columns=['user_id', 'item_id', 'rating'])
    
    # 保存示例数据
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    interactions.to_csv(os.path.join(data_dir, 'interactions.csv'), index=False)
    
    # 预处理数据
    preprocessor = DataPreprocessor(data_dir)
    interactions = preprocessor.load_data()
    train_data, val_data, test_data = preprocessor.split_data(interactions)
    
    # 创建负样本
    train_data_with_neg = preprocessor.create_negatives(train_data)
    val_data_with_neg = preprocessor.create_negatives(val_data)
    test_data_with_neg = preprocessor.create_negatives(test_data)
    
    # 保存处理后的数据
    output_dir = os.path.join(data_dir, 'processed')
    preprocessor.save_processed_data(train_data_with_neg, val_data_with_neg, test_data_with_neg, output_dir)
    
    print(f"数据预处理完成！")
    print(f"特征维度: {preprocessor.get_feature_dimensions()}")
    print(f"训练集大小: {len(train_data_with_neg)}")
    print(f"验证集大小: {len(val_data_with_neg)}")
    print(f"测试集大小: {len(test_data_with_neg)}")