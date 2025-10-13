import os
import json
import pandas as pd
import numpy as np
from data_preprocessor import DataPreprocessor
from two_tower_model import TwoTowerModel
import tensorflow as tf
import matplotlib.pyplot as plt

class TwoTowerRecallSystem:
    def __init__(self, config_path=None):
        """
        初始化双塔召回系统
        :param config_path: 配置文件路径
        """
        # 默认配置
        self.default_config = {
            'data': {
                'interactions_file': 'interactions.csv',
                'user_features_file': None,
                'item_features_file': None,
                'test_size': 0.2,
                'val_size': 0.1,
                'num_negatives': 4
            },
            'model': {
                'embedding_dim': 64,
                'latent_dim': 32,
                'user_tower_layers': [128, 64],
                'item_tower_layers': [128, 64],
                'dropout_rate': 0.3,
                'use_batch_norm': True,
                'learning_rate': 0.001,
                'batch_size': 256,
                'epochs': 10
            },
            'paths': {
                'data_dir': '../data',
                'processed_data_dir': '../data/processed',
                'model_dir': '../models'
            }
        }
        
        # 加载配置文件（如果提供）
        self.config = self.default_config
        if config_path:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                self._merge_config(self.config, user_config)
        
        # 创建目录
        for dir_path in self.config['paths'].values():
            abs_dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), dir_path.lstrip('../'))
            if not os.path.exists(abs_dir_path):
                os.makedirs(abs_dir_path)
        
        # 初始化组件
        self.preprocessor = None
        self.model = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def _merge_config(self, base_config, user_config):
        """
        合并配置
        :param base_config: 基础配置
        :param user_config: 用户配置
        """
        for key, value in user_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def prepare_data(self, generate_sample_data=False):
        """
        准备数据
        :param generate_sample_data: 是否生成示例数据
        """
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), self.config['paths']['data_dir'].lstrip('../'))
        processed_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), self.config['paths']['processed_data_dir'].lstrip('../'))
        
        # 检查是否已经有预处理后的数据
        processed_files = ['train_data.csv', 'val_data.csv', 'test_data.csv']
        if all(os.path.exists(os.path.join(processed_data_dir, f)) for f in processed_files) and not generate_sample_data:
            print("发现已有的预处理数据，直接加载...")
            self.train_data = pd.read_csv(os.path.join(processed_data_dir, 'train_data.csv'))
            self.val_data = pd.read_csv(os.path.join(processed_data_dir, 'val_data.csv'))
            self.test_data = pd.read_csv(os.path.join(processed_data_dir, 'test_data.csv'))
            return
        
        # 创建数据预处理器
        self.preprocessor = DataPreprocessor(data_dir)
        
        # 如果需要生成示例数据
        if generate_sample_data or not os.path.exists(os.path.join(data_dir, self.config['data']['interactions_file'])):
            print("生成示例数据...")
            self._generate_sample_data(data_dir)
        
        # 加载数据
        print("加载数据...")
        interactions = self.preprocessor.load_data(
            interactions_file=self.config['data']['interactions_file'],
            user_features_file=self.config['data']['user_features_file'],
            item_features_file=self.config['data']['item_features_file']
        )
        
        # 分割数据
        print("分割数据...")
        train_data, val_data, test_data = self.preprocessor.split_data(
            interactions,
            test_size=self.config['data']['test_size'],
            val_size=self.config['data']['val_size']
        )
        
        # 创建负样本
        print("创建负样本...")
        self.train_data = self.preprocessor.create_negatives(train_data, num_negatives=self.config['data']['num_negatives'])
        self.val_data = self.preprocessor.create_negatives(val_data, num_negatives=self.config['data']['num_negatives'])
        self.test_data = self.preprocessor.create_negatives(test_data, num_negatives=self.config['data']['num_negatives'])
        
        # 保存处理后的数据
        print("保存预处理数据...")
        self.preprocessor.save_processed_data(self.train_data, self.val_data, self.test_data, processed_data_dir)
        
        print("数据准备完成！")
        print(f"训练集大小: {len(self.train_data)}")
        print(f"验证集大小: {len(self.val_data)}")
        print(f"测试集大小: {len(self.test_data)}")
    
    def _generate_sample_data(self, data_dir):
        """
        生成示例数据
        :param data_dir: 数据目录
        """
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
        interactions.to_csv(os.path.join(data_dir, self.config['data']['interactions_file']), index=False)
        print(f"示例数据已保存到: {os.path.join(data_dir, self.config['data']['interactions_file'])}")
    
    def build_model(self):
        """
        构建模型
        """
        # 获取特征维度信息
        if self.preprocessor:
            feature_dims = self.preprocessor.get_feature_dimensions()
        else:
            # 从数据中获取特征维度
            num_users = self.train_data['user_id'].nunique()
            num_items = self.train_data['item_id'].nunique()
            feature_dims = {'num_users': num_users, 'num_items': num_items}
        
        # 合并模型配置
        model_config = {
            **self.config['model'],
            **feature_dims
        }
        
        # 创建并构建模型
        self.model = TwoTowerModel(model_config)
        self.model.build_model()
        
        # 打印模型结构
        print("模型结构:")
        self.model.model.summary()
    
    def train_model(self):
        """
        训练模型
        """
        if not self.model:
            self.build_model()
        
        # 准备回调函数
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
        ]
        
        # 训练模型
        print("开始训练模型...")
        history = self.model.train(
            self.train_data,
            self.val_data,
            batch_size=self.config['model']['batch_size'],
            epochs=self.config['model']['epochs'],
            callbacks=callbacks
        )
        
        # 可视化训练过程
        self._plot_training_history(history)
        
        return history
    
    def _plot_training_history(self, history):
        """
        可视化训练历史
        :param history: 训练历史
        """
        # 创建图形
        plt.figure(figsize=(12, 5))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='训练损失')
        plt.plot(history.history['val_loss'], label='验证损失')
        plt.title('模型损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        
        # 绘制准确率曲线
        if 'accuracy' in history.history:
            plt.subplot(1, 2, 2)
            plt.plot(history.history['accuracy'], label='训练准确率')
            plt.plot(history.history['val_accuracy'], label='验证准确率')
            plt.title('模型准确率')
            plt.xlabel('轮次')
            plt.ylabel('准确率')
            plt.legend()
        
        plt.tight_layout()
        
        # 保存图像
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), self.config['paths']['model_dir'].lstrip('../'))
        plt.savefig(os.path.join(model_dir, 'training_history.png'))
        print(f"训练历史图像已保存到: {os.path.join(model_dir, 'training_history.png')}")
        
        # 显示图像
        plt.show()
    
    def evaluate_model(self):
        """
        评估模型
        """
        if not self.model:
            print("请先训练模型")
            return None
        
        print("评估模型性能...")
        evaluation_results = self.model.evaluate(self.test_data, batch_size=self.config['model']['batch_size'])
        
        # 保存评估结果
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), self.config['paths']['model_dir'].lstrip('../'))
        with open(os.path.join(model_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"评估结果已保存到: {os.path.join(model_dir, 'evaluation_results.json')}")
        
        return evaluation_results
    
    def save_model(self):
        """
        保存模型
        """
        if not self.model:
            print("请先训练模型")
            return
        
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), self.config['paths']['model_dir'].lstrip('../'))
        self.model.save_model(model_dir)
    
    def load_model(self):
        """
        加载模型
        """
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), self.config['paths']['model_dir'].lstrip('../'))
        self.model = TwoTowerModel.load_model(model_dir)
        
        # 加载配置
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            model_config = json.load(f)
        
        # 更新配置
        self.config['model'] = {**self.config['model'], **model_config}
        
        return self.model
    
    def recommend_items(self, user_id, top_k=10):
        """
        为指定用户推荐物品
        :param user_id: 用户ID
        :param top_k: 推荐的物品数量
        :return: 推荐物品列表和对应的相似度分数
        """
        if not self.model:
            print("请先加载或训练模型")
            return None
        
        # 获取所有物品的ID
        all_items = self.test_data['item_id'].unique()
        
        # 获取用户嵌入向量
        user_embedding = self.model.get_user_embeddings([user_id])[0]
        
        # 获取所有物品的嵌入向量
        item_embeddings = self.model.get_item_embeddings(all_items)
        
        # 计算用户向量与所有物品向量的相似度
        similarities = np.dot(item_embeddings, user_embedding)
        
        # 获取相似度最高的Top-K个物品
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_items = [all_items[i] for i in top_indices]
        top_scores = [similarities[i] for i in top_indices]
        
        return list(zip(top_items, top_scores))
    
    def run_pipeline(self, generate_sample_data=False):
        """
        运行完整的召回系统流程
        :param generate_sample_data: 是否生成示例数据
        """
        try:
            # 1. 准备数据
            self.prepare_data(generate_sample_data)
            
            # 2. 构建模型
            self.build_model()
            
            # 3. 训练模型
            self.train_model()
            
            # 4. 评估模型
            self.evaluate_model()
            
            # 5. 保存模型
            self.save_model()
            
            print("双塔召回系统运行完成！")
            
            # 6. 示例推荐
            sample_user_id = np.random.choice(self.test_data['user_id'].unique())
            print(f"\n为用户 {sample_user_id} 推荐的Top-10物品:")
            recommendations = self.recommend_items(sample_user_id)
            for item_id, score in recommendations:
                print(f"物品ID: {item_id}, 相似度分数: {score:.4f}")
                
        except Exception as e:
            print(f"运行出错: {str(e)}")
            import traceback
            traceback.print_exc()

# 示例用法
if __name__ == '__main__':
    # 创建配置文件路径
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'config.json')
    
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        print(f"配置文件 {config_path} 不存在，使用默认配置...")
        config_path = None
    
    # 创建并运行双塔召回系统
    recall_system = TwoTowerRecallSystem(config_path)
    
    # 运行完整流程（设置generate_sample_data=True生成示例数据）
    recall_system.run_pipeline(generate_sample_data=True)