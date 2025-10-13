import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, BatchNormalization, Concatenate
import numpy as np
import os
import json

class TwoTowerModel:
    def __init__(self, config):
        """
        初始化双塔模型
        :param config: 模型配置字典
        """
        self.config = config
        self.model = None
        self.user_tower = None
        self.item_tower = None
        
    def build_model(self):
        """
        构建双塔模型结构
        """
        # 1. 用户塔 (User Tower)
        # 用户ID输入
        user_id_input = Input(shape=(1,), name='user_id')
        # 用户嵌入层
        user_embedding = Embedding(
            input_dim=self.config['num_users'],
            output_dim=self.config['embedding_dim'],
            name='user_embedding'
        )(user_id_input)
        # 扁平化嵌入向量
        user_embedding_flat = tf.keras.layers.Flatten(name='user_embedding_flat')(user_embedding)
        
        # 用户塔全连接层
        user_layer = user_embedding_flat
        for i, units in enumerate(self.config['user_tower_layers']):
            user_layer = Dense(units, activation='relu', name=f'user_dense_{i}')(user_layer)
            if self.config['use_batch_norm']:
                user_layer = BatchNormalization(name=f'user_bn_{i}')(user_layer)
            if self.config['dropout_rate'] > 0:
                user_layer = Dropout(self.config['dropout_rate'], name=f'user_dropout_{i}')(user_layer)
        
        # 用户向量输出
        user_output = Dense(self.config['latent_dim'], activation='relu', name='user_output')(user_layer)
        
        # 2. 物品塔 (Item Tower)
        # 物品ID输入
        item_id_input = Input(shape=(1,), name='item_id')
        # 物品嵌入层
        item_embedding = Embedding(
            input_dim=self.config['num_items'],
            output_dim=self.config['embedding_dim'],
            name='item_embedding'
        )(item_id_input)
        # 扁平化嵌入向量
        item_embedding_flat = tf.keras.layers.Flatten(name='item_embedding_flat')(item_embedding)
        
        # 物品塔全连接层
        item_layer = item_embedding_flat
        for i, units in enumerate(self.config['item_tower_layers']):
            item_layer = Dense(units, activation='relu', name=f'item_dense_{i}')(item_layer)
            if self.config['use_batch_norm']:
                item_layer = BatchNormalization(name=f'item_bn_{i}')(item_layer)
            if self.config['dropout_rate'] > 0:
                item_layer = Dropout(self.config['dropout_rate'], name=f'item_dropout_{i}')(item_layer)
        
        # 物品向量输出
        item_output = Dense(self.config['latent_dim'], activation='relu', name='item_output')(item_layer)
        
        # 3. 计算用户向量和物品向量的点积相似度
        dot_product = tf.keras.layers.Dot(axes=1, normalize=True)([user_output, item_output])
        output = Dense(1, activation='sigmoid', name='output')(dot_product)
        
        # 4. 构建完整模型
        self.model = Model(inputs=[user_id_input, item_id_input], outputs=output)
        
        # 5. 保存用户塔和物品塔的单独模型，用于离线推理
        self.user_tower = Model(inputs=user_id_input, outputs=user_output)
        self.item_tower = Model(inputs=item_id_input, outputs=item_output)
        
        # 6. 编译模型
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return self.model
    
    def train(self, train_data, val_data, batch_size=None, epochs=None, callbacks=None):
        """
        训练模型
        :param train_data: 训练数据
        :param val_data: 验证数据
        :param batch_size: 批次大小
        :param epochs: 训练轮数
        :param callbacks: 回调函数列表
        :return: 训练历史
        """
        if batch_size is None:
            batch_size = self.config.get('batch_size', 256)
        if epochs is None:
            epochs = self.config.get('epochs', 10)
        
        # 准备训练数据
        train_user_ids = train_data['user_id'].values
        train_item_ids = train_data['item_id'].values
        train_labels = train_data['label'].values
        
        # 准备验证数据
        val_user_ids = val_data['user_id'].values
        val_item_ids = val_data['item_id'].values
        val_labels = val_data['label'].values
        
        # 训练模型
        history = self.model.fit(
            x=[train_user_ids, train_item_ids],
            y=train_labels,
            validation_data=([val_user_ids, val_item_ids], val_labels),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            shuffle=True
        )
        
        return history
    
    def evaluate(self, test_data, batch_size=None):
        """
        评估模型
        :param test_data: 测试数据
        :param batch_size: 批次大小
        :return: 评估结果
        """
        if batch_size is None:
            batch_size = self.config.get('batch_size', 256)
        
        # 准备测试数据
        test_user_ids = test_data['user_id'].values
        test_item_ids = test_data['item_id'].values
        test_labels = test_data['label'].values
        
        # 评估模型
        loss, accuracy, auc = self.model.evaluate(
            x=[test_user_ids, test_item_ids],
            y=test_labels,
            batch_size=batch_size,
            verbose=1
        )
        
        print(f"测试集评估结果: 损失={loss:.4f}, 准确率={accuracy:.4f}, AUC={auc:.4f}")
        
        return {'loss': loss, 'accuracy': accuracy, 'auc': auc}
    
    def predict(self, user_ids, item_ids, batch_size=None):
        """
        预测用户对物品的交互概率
        :param user_ids: 用户ID列表
        :param item_ids: 物品ID列表
        :param batch_size: 批次大小
        :return: 预测概率
        """
        if batch_size is None:
            batch_size = self.config.get('batch_size', 256)
        
        # 预测
        predictions = self.model.predict(
            x=[np.array(user_ids), np.array(item_ids)],
            batch_size=batch_size,
            verbose=0
        )
        
        return predictions.flatten()
    
    def get_user_embeddings(self, user_ids, batch_size=None):
        """
        获取用户嵌入向量
        :param user_ids: 用户ID列表
        :param batch_size: 批次大小
        :return: 用户嵌入向量
        """
        if batch_size is None:
            batch_size = self.config.get('batch_size', 256)
        
        # 获取用户嵌入向量
        user_embeddings = self.user_tower.predict(
            x=np.array(user_ids),
            batch_size=batch_size,
            verbose=0
        )
        
        return user_embeddings
    
    def get_item_embeddings(self, item_ids, batch_size=None):
        """
        获取物品嵌入向量
        :param item_ids: 物品ID列表
        :param batch_size: 批次大小
        :return: 物品嵌入向量
        """
        if batch_size is None:
            batch_size = self.config.get('batch_size', 256)
        
        # 获取物品嵌入向量
        item_embeddings = self.item_tower.predict(
            x=np.array(item_ids),
            batch_size=batch_size,
            verbose=0
        )
        
        return item_embeddings
    
    def save_model(self, model_dir):
        """
        保存模型
        :param model_dir: 模型保存目录
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # 保存完整模型
        self.model.save(os.path.join(model_dir, 'complete_model'))
        
        # 保存用户塔和物品塔
        self.user_tower.save(os.path.join(model_dir, 'user_tower'))
        self.item_tower.save(os.path.join(model_dir, 'item_tower'))
        
        # 保存配置文件
        with open(os.path.join(model_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"模型已保存到: {model_dir}")
    
    @classmethod
    def load_model(cls, model_dir):
        """
        加载模型
        :param model_dir: 模型保存目录
        :return: TwoTowerModel实例
        """
        # 加载配置文件
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # 创建模型实例
        model = cls(config)
        
        # 加载完整模型
        model.model = tf.keras.models.load_model(os.path.join(model_dir, 'complete_model'))
        
        # 加载用户塔和物品塔
        model.user_tower = tf.keras.models.load_model(os.path.join(model_dir, 'user_tower'))
        model.item_tower = tf.keras.models.load_model(os.path.join(model_dir, 'item_tower'))
        
        print(f"模型已从: {model_dir} 加载")
        
        return model

# 示例用法
if __name__ == '__main__':
    # 假设我们已经有了预处理后的数据
    import pandas as pd
    import os
    
    # 设置数据目录
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed')
    
    # 加载预处理后的数据
    try:
        train_data = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
        val_data = pd.read_csv(os.path.join(data_dir, 'val_data.csv'))
        test_data = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
    except FileNotFoundError:
        print("请先运行data_preprocessor.py生成预处理数据")
        exit()
    
    # 模型配置
    config = {
        'num_users': train_data['user_id'].nunique(),
        'num_items': train_data['item_id'].nunique(),
        'embedding_dim': 64,
        'latent_dim': 32,
        'user_tower_layers': [128, 64],
        'item_tower_layers': [128, 64],
        'dropout_rate': 0.3,
        'use_batch_norm': True,
        'learning_rate': 0.001,
        'batch_size': 256,
        'epochs': 10
    }
    
    # 创建并构建模型
    two_tower = TwoTowerModel(config)
    two_tower.build_model()
    
    # 打印模型结构
    two_tower.model.summary()
    
    # 准备回调函数
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
    ]
    
    # 训练模型
    print("开始训练模型...")
    history = two_tower.train(train_data, val_data, callbacks=callbacks)
    
    # 评估模型
    print("评估模型性能...")
    evaluation_results = two_tower.evaluate(test_data)
    
    # 保存模型
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    two_tower.save_model(model_dir)
    
    # 示例：为用户推荐物品
    def recommend_items(user_id, model, all_items, top_k=10):
        """为指定用户推荐Top-K个物品"""
        # 获取用户嵌入向量
        user_embedding = model.get_user_embeddings([user_id])[0]
        
        # 获取所有物品的嵌入向量
        item_embeddings = model.get_item_embeddings(all_items)
        
        # 计算用户向量与所有物品向量的相似度
        similarities = np.dot(item_embeddings, user_embedding)
        
        # 获取相似度最高的Top-K个物品
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_items = [all_items[i] for i in top_indices]
        top_scores = [similarities[i] for i in top_indices]
        
        return list(zip(top_items, top_scores))
    
    # 为随机选择的用户推荐物品
    sample_user_id = np.random.choice(test_data['user_id'].unique())
    all_items = test_data['item_id'].unique()
    
    print(f"\n为用户 {sample_user_id} 推荐的Top-10物品:")
    recommendations = recommend_items(sample_user_id, two_tower, all_items)
    for item_id, score in recommendations:
        print(f"物品ID: {item_id}, 相似度分数: {score:.4f}")