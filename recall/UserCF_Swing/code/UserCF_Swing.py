import numpy as np
import pandas as pd
import os
import random
import numpy as np  # 添加numpy库导入
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import time
import pickle 
from datetime import datetime


# ----------------------
# 1. 数据生成器类
# ----------------------
class DataGenerator:
    def __init__(self):
        # 设置数据规模参数
        self.num_users = 500        # 用户数量
        self.num_items = 1000       # 物品数量
        self.num_interactions = 5000 # 交互记录数量
        
        # 用户特征参数
        self.user_age_range = (18, 65)
        self.user_genders = ['男', '女']
        
        # 物品特征参数
        self.item_categories = ['电子产品', '服装', '食品', '图书', '家居', '运动', '美妆', '玩具']
        self.item_prices = {
            '电子产品': (1000, 5000),
            '服装': (100, 1000),
            '食品': (10, 200),
            '图书': (20, 200),
            '家居': (50, 2000),
            '运动': (100, 1500),
            '美妆': (50, 800),
            '玩具': (30, 500)
        }
        
        # 交互类型参数
        self.interaction_types = ['点击', '收藏', '加入购物车', '购买']
        self.interaction_weights = {
            '点击': 1,
            '收藏': 2,
            '加入购物车': 3,
            '购买': 5
        }
        
        # 设置随机种子，保证数据可复现
        random.seed(42)
        np.random.seed(42)  # 修复：现在已经正确导入了numpy
    
    def generate_users(self) -> pd.DataFrame:
        """生成用户数据"""
        user_ids = range(1000, 1000 + self.num_users)
        users = []
        
        for user_id in user_ids:
            users.append({
                'user_id': user_id,
                'age': random.randint(self.user_age_range[0], self.user_age_range[1]),
                'gender': random.choice(self.user_genders)
            })
        
        return pd.DataFrame(users)
    
    def generate_items(self) -> pd.DataFrame:
        """生成物品数据"""
        item_ids = range(2000, 2000 + self.num_items)
        items = []
        
        for item_id in item_ids:
            category = random.choice(self.item_categories)
            min_price, max_price = self.item_prices[category]
            
            # 生成物品名称（类别+随机数字）
            item_name = f"{category}_{random.randint(100, 999)}"
            
            items.append({
                'item_id': item_id,
                'item_name': item_name,
                'category': category,
                'price': round(random.uniform(min_price, max_price), 2)
            })
        
        return pd.DataFrame(items)
    
    def generate_interactions(self, users: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
        """生成用户-物品交互数据"""
        interactions = []
        user_ids = users['user_id'].tolist()
        item_ids = items['item_id'].tolist()
        
        # 创建物品-类别映射，用于构建用户兴趣偏好
        item_to_category = dict(zip(items['item_id'], items['category']))
        
        # 为每个用户设置偏好类别（增加协同过滤的效果）
        user_preferences = {}
        for user_id in user_ids:
            # 每个用户有2-3个偏好类别
            num_preferred = random.randint(2, 3)
            preferred_categories = random.sample(self.item_categories, num_preferred)
            user_preferences[user_id] = preferred_categories
        
        # 生成交互记录
        for _ in range(self.num_interactions):
            user_id = random.choice(user_ids)
            preferred_categories = user_preferences[user_id]
            
            # 70%的概率选择用户偏好类别
            if random.random() < 0.7 and preferred_categories:
                category = random.choice(preferred_categories)
                # 从该类别中选择物品
                category_items = [item for item in item_ids if item_to_category[item] == category]
                if category_items:
                    item_id = random.choice(category_items)
                else:
                    item_id = random.choice(item_ids)
            else:
                item_id = random.choice(item_ids)
            
            # 生成交互类型，购买概率较低
            interaction_type = random.choices(
                self.interaction_types,
                weights=[0.5, 0.2, 0.2, 0.1],  # 点击概率最高，购买最低
                k=1
            )[0]
            
            # 生成交互时间（过去30天内的随机时间）
            days_ago = random.randint(0, 29)
            interaction_time = (datetime.now() - pd.Timedelta(days=days_ago)).strftime('%Y-%m-%d %H:%M:%S')
            
            interactions.append({
                'user_id': user_id,
                'item_id': item_id,
                'interaction_type': interaction_type,
                'interaction_time': interaction_time
            })
        
        # 去重（避免同一用户对同一物品的重复交互）
        interactions_df = pd.DataFrame(interactions)
        interactions_df = interactions_df.drop_duplicates(subset=['user_id', 'item_id'])
        
        # 修复：确保示例用户1003有交互历史
        if not interactions_df[interactions_df['user_id'] == 1003].empty:
            # 如果用户1003已有交互，就不做处理
            pass
        else:
            # 如果用户1003没有交互历史，为其生成一些交互
            for _ in range(3):  # 生成3条交互记录
                category = random.choice(self.item_categories)
                category_items = [item for item in item_ids if item_to_category[item] == category]
                if category_items:
                    item_id = random.choice(category_items)
                    interaction_type = random.choices(
                        self.interaction_types,
                        weights=[0.5, 0.2, 0.2, 0.1],
                        k=1
                    )[0]
                    days_ago = random.randint(0, 29)
                    interaction_time = (datetime.now() - pd.Timedelta(days=days_ago)).strftime('%Y-%m-%d %H:%M:%S')
                    
                    interactions_df = pd.concat([
                        interactions_df,
                        pd.DataFrame([{
                            'user_id': 1003,
                            'item_id': item_id,
                            'interaction_type': interaction_type,
                            'interaction_time': interaction_time
                        }])
                    ])
        
        return interactions_df
    
    def generate_and_save_data(self, output_dir: str) -> None:
        """生成并保存所有数据到CSV文件"""
        start_time = time.time()
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成数据
        print("📊 开始生成数据...")
        users_df = self.generate_users()
        items_df = self.generate_items()
        interactions_df = self.generate_interactions(users_df, items_df)
        
        # 保存数据
        users_df.to_csv(os.path.join(output_dir, 'user_table.csv'), index=False)
        items_df.to_csv(os.path.join(output_dir, 'item_table.csv'), index=False)
        interactions_df.to_csv(os.path.join(output_dir, 'interaction_table.csv'), index=False)
        
        print(f"✅ 数据生成完成！")
        print(f"📁 保存目录：{output_dir}")
        print(f"📊 数据规模：{len(users_df)}用户 | {len(items_df)}物品 | {len(interactions_df)}交互记录")
        print(f"⏱️  耗时：{time.time() - start_time:.2f}秒")

# ----------------------
# 2. UserCF+Swing推荐器类
# ----------------------
class UserCFSwingRecommender:
    def __init__(self, data_dir: str, cache_dir: Optional[str] = None, load_from_cache: bool = False):
        self.data_dir = data_dir
        self.user_df = None
        self.item_df = None
        self.interaction_df = None
        self.user_similarity = None
        self.item_to_users = None  # 物品到用户的倒排表
        
        # 设置缓存目录，默认为data_dir下的cache文件夹
        self.cache_dir = cache_dir if cache_dir else os.path.join(data_dir, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 先尝试加载缓存
        cache_loaded = False
        if load_from_cache:
            cache_loaded = self.load_params()
        
        # 无论是否加载缓存成功，都需要加载基础数据（用户和物品信息）
        self._load_basic_data()
        
        # 如果缓存未加载或加载失败，则构建物品到用户的倒排表
        if not cache_loaded:
            # 加载完整交互数据
            interaction_path = os.path.join(self.data_dir, 'interaction_table.csv')
            if os.path.exists(interaction_path):
                self.interaction_df = pd.read_csv(interaction_path)
            # 构建物品到用户的倒排表
            self._build_item_to_users()
    
    def _load_basic_data(self) -> None:
        """加载用户和物品的基础信息"""
        start_time = time.time()
        user_path = os.path.join(self.data_dir, 'user_table.csv')
        item_path = os.path.join(self.data_dir, 'item_table.csv')
        
        # 检查数据文件是否存在
        missing_files = [f for f in [user_path, item_path] if not os.path.exists(f)]
        if missing_files:
            raise FileNotFoundError(f"❌ 缺失数据文件：{', '.join(missing_files)}\n请先运行数据生成逻辑！")
        
        try:
            self.user_df = pd.read_csv(user_path)
            self.item_df = pd.read_csv(item_path)
            
            print(f"📥 基础数据加载完成（耗时：{time.time() - start_time:.2f}秒）")
            print(f"📥 数据规模：{len(self.user_df)}用户 | {len(self.item_df)}物品")
        except Exception as e:
            raise RuntimeError(f"❌ 数据加载失败：{str(e)}") from e
    
    def _load_data(self) -> None:
        """加载用户、物品和交互数据"""
        # 首先加载基础数据
        self._load_basic_data()
        
        # 然后加载交互数据
        interaction_path = os.path.join(self.data_dir, 'interaction_table.csv')
        if os.path.exists(interaction_path):
            self.interaction_df = pd.read_csv(interaction_path)
            print(f"📥 交互数据加载完成，共{len(self.interaction_df)}条记录")
    
    def _build_item_to_users(self) -> None:
        """构建物品到用户的倒排表"""
        if self.interaction_df is None:
            raise ValueError("❌ 未加载交互数据")
            
        start_time = time.time()
        self.item_to_users = defaultdict(list)
        
        # 按物品分组，记录交互过该物品的所有用户
        for _, row in self.interaction_df.iterrows():
            item_id = row['item_id']
            user_id = row['user_id']
            if user_id not in self.item_to_users[item_id]:
                self.item_to_users[item_id].append(user_id)
        
        print(f"🔧 物品到用户倒排表构建完成（耗时：{time.time() - start_time:.2f}秒）")
        print(f"🔧 覆盖物品数量：{len(self.item_to_users)}")
    
    def calculate_user_similarity(self, use_weights: bool = False) -> None:
        """使用Swing算法计算用户相似度
        
        Args:
            use_weights: 是否使用交互权重（购买>收藏>点击）
        """
        if self.item_to_users is None:
            raise ValueError("❌ 未构建物品到用户倒排表")
        
        start_time = time.time()
        self.user_similarity = defaultdict(dict)
        
        # 遍历每个物品的用户列表
        for item_id, users in self.item_to_users.items():
            # 获取交互过该物品的用户数量（用于热门惩罚）
            user_count = len(users)
            
            # 计算热门惩罚项
            penalty = 1.0 / np.log(1 + user_count)
            
            # 遍历用户对，计算相似度贡献
            for i in range(len(users)):
                for j in range(i + 1, len(users)):
                    u = users[i]
                    v = users[j]
                    
                    # 计算贡献值
                    contribution = penalty
                    
                    # 如果使用交互权重，需要获取每个用户对该物品的交互权重
                    if use_weights and self.interaction_df is not None:
                        # 查找用户u和v对该物品的交互类型
                        u_interaction = self.interaction_df[
                            (self.interaction_df['user_id'] == u) & 
                            (self.interaction_df['item_id'] == item_id)
                        ]
                        v_interaction = self.interaction_df[
                            (self.interaction_df['user_id'] == v) & 
                            (self.interaction_df['item_id'] == item_id)
                        ]
                        
                        # 获取交互权重
                        if not u_interaction.empty and not v_interaction.empty:
                            u_weight = self._get_interaction_weight(u_interaction['interaction_type'].iloc[0])
                            v_weight = self._get_interaction_weight(v_interaction['interaction_type'].iloc[0])
                            contribution *= (u_weight * v_weight)
                    
                    # 累加到用户相似度矩阵
                    if v not in self.user_similarity[u]:
                        self.user_similarity[u][v] = 0.0
                    if u not in self.user_similarity[v]:
                        self.user_similarity[v][u] = 0.0
                    
                    self.user_similarity[u][v] += contribution
                    self.user_similarity[v][u] += contribution
        
        print(f"🔍 用户相似度计算完成（耗时：{time.time() - start_time:.2f}秒）")
        print(f"🔍 计算了 {len(self.user_similarity)} 个用户的相似度")
    
    def _get_interaction_weight(self, interaction_type: str) -> int:
        """获取交互类型的权重"""
        weight_map = {
            "点击": 1,
            "收藏": 2,
            "加入购物车": 3,
            "购买": 5  # 购买权重最高
        }
        return weight_map.get(interaction_type, 0)
    
    def recommend_items(self, user_id: int, top_n: int = 10, k_similar_users: int = 50) -> List[Tuple[int, float]]:
        """为目标用户推荐物品
        
        Args:
            user_id: 目标用户ID
            top_n: 推荐物品数量
            k_similar_users: 参考的相似用户数量
        """
        if self.user_similarity is None:
            raise ValueError("❌ 未计算用户相似度")
        
        # 获取用户已交互的物品集合
        user_items = set()
        if self.interaction_df is not None:
            user_items = set(self.interaction_df[self.interaction_df['user_id'] == user_id]['item_id'])
        
        if not user_items:
            print(f"⚠️ 用户{user_id}无交互历史，无法推荐")
            return []
        
        # 获取目标用户的相似用户（按相似度排序）
        similar_users = sorted(
            self.user_similarity.get(user_id, {}).items(),
            key=lambda x: x[1],
            reverse=True
        )[:k_similar_users]
        
        if not similar_users:
            print(f"⚠️ 未找到用户{user_id}的相似用户")
            return []
        
        # 计算候选物品分数
        item_scores = defaultdict(float)
        for similar_user, similarity in similar_users:
            # 获取相似用户交互过的物品
            if self.interaction_df is not None:
                similar_user_items = self.interaction_df[self.interaction_df['user_id'] == similar_user]
                
                # 遍历相似用户的物品，计算分数
                for _, row in similar_user_items.iterrows():
                    item_id = row['item_id']
                    # 过滤掉用户已交互的物品
                    if item_id not in user_items:
                        # 可以选择加上交互权重
                        weight = self._get_interaction_weight(row['interaction_type'])
                        item_scores[item_id] += similarity * weight
        
        # 按分数排序，返回Top-N物品
        return sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def evaluate(self, test_user_ids: Optional[List[int]] = None, top_n: int = 10) -> Dict[str, float]:
        """评估推荐系统效果"""
        if self.user_similarity is None:
            raise ValueError("❌ 未计算用户相似度")
        
        if self.interaction_df is None:
            raise ValueError("❌ 未加载交互数据，无法评估")
        
        # 选择测试用户（默认选20%有交互的用户）
        active_users = self.interaction_df["user_id"].unique()
        if test_user_ids is None:
            test_size = max(1, int(len(active_users) * 0.2))
            test_user_ids = random.sample(list(active_users), test_size)
        
        total_recall = 0.0
        total_precision = 0.0
        valid_users = 0
        
        for user_id in test_user_ids:
            # 真实正样本：用户交互过的物品
            real_items = set(self.interaction_df[self.interaction_df["user_id"] == user_id]["item_id"])
            if len(real_items) < 2:
                continue  # 交互太少，评估无意义
            
            # 推荐物品：模型输出的候选
            recommended_items = set([item_id for item_id, _ in self.recommend_items(user_id, top_n)])
            if not recommended_items:
                continue
            
            # 命中物品：推荐中包含的真实正样本
            hit_items = recommended_items & real_items
            valid_users += 1
            
            # 计算召回率和精确率
            total_recall += len(hit_items) / len(real_items) if real_items else 0.0
            total_precision += len(hit_items) / len(recommended_items) if recommended_items else 0.0
        
        # 返回平均指标
        return {
            "测试用户数": valid_users,
            "平均召回率": round(total_recall / valid_users, 4) if valid_users else 0.0,
            "平均精确率": round(total_precision / valid_users, 4) if valid_users else 0.0
        }
    
    def print_similar_users(self, user_id: int, top_k: int = 5) -> None:
        """打印用户的相似用户"""
        if self.user_similarity is None:
            raise ValueError("❌ 未计算用户相似度")
        
        if self.user_df is None:
            raise ValueError("❌ 未加载用户数据")
        
        similar_users = sorted(
            self.user_similarity.get(user_id, {}).items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        print(f"\n👥 用户{user_id}的Top{top_k}相似用户：")
        if not similar_users:
            print("  - 未找到相似用户")
        else:
            for i, (similar_user_id, similarity) in enumerate(similar_users, 1):
                # 获取相似用户信息
                similar_user_info = self.user_df[self.user_df['user_id'] == similar_user_id]
                age = similar_user_info['age'].iloc[0] if not similar_user_info.empty else '未知'
                gender = similar_user_info['gender'].iloc[0] if not similar_user_info.empty else '未知'
                print(f"  {i}. 用户{similar_user_id}（年龄：{age}，性别：{gender}）：相似度 {similarity:.4f}")
    
    def print_recommendations(self, user_id: int, top_n: int = 5) -> None:
        """打印用户的推荐结果"""
        if self.user_df is None:
            raise ValueError("❌ 未加载用户数据")
        
        if self.item_df is None:
            raise ValueError("❌ 未加载物品数据")
        
        # 1. 打印用户交互历史（前5条）
        print(f"\n👤 用户{user_id}的交互历史（前5条）：")
        if self.interaction_df is not None:
            user_interactions = self.interaction_df[self.interaction_df["user_id"] == user_id].head(5)
            if user_interactions.empty:
                print("  - 无交互历史")
            else:
                for _, row in user_interactions.iterrows():
                    item_info = self.item_df[self.item_df['item_id'] == row['item_id']]
                    item_name = item_info['item_name'].iloc[0] if not item_info.empty else f"物品{row['item_id']}"
                    item_category = item_info['category'].iloc[0] if not item_info.empty else '未知'
                    print(f"  - {item_name}（ID：{row['item_id']}，类别：{item_category}）：{row['interaction_type']}")
        else:
            print("  - 交互数据未加载")
        
        # 2. 打印推荐结果
        recommendations = self.recommend_items(user_id, top_n)
        print(f"\n🎯 为用户{user_id}推荐的Top{top_n}物品：")
        if not recommendations:
            print("  - 无推荐物品")
        else:
            for i, (item_id, score) in enumerate(recommendations, 1):
                item_info = self.item_df[self.item_df['item_id'] == item_id]
                item_name = item_info['item_name'].iloc[0] if not item_info.empty else f"物品{item_id}"
                item_category = item_info['category'].iloc[0] if not item_info.empty else '未知'
                item_price = item_info['price'].iloc[0] if not item_info.empty else '未知'
                print(f"  {i}. {item_name}（ID：{item_id}）")
                print(f"     - 推荐分数：{score:.4f} | 类别：{item_category} | 价格：¥{item_price}")
    
    def save_params(self) -> None:
        """持久化保存推荐器的关键参数"""
        start_time = time.time()
        
        # 要保存的参数
        params_to_save = {
            'user_similarity': self.user_similarity,
            'item_to_users': self.item_to_users
            # 注意：用户和物品基础数据不保存到缓存，每次运行时重新加载
        }
        
        # 构建缓存文件路径
        cache_file = os.path.join(self.cache_dir, f'usercf_swing_params_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(params_to_save, f)
            
            print(f"💾 参数持久化完成（耗时：{time.time() - start_time:.2f}秒）")
            print(f"💾 缓存文件路径：{cache_file}")
        except Exception as e:
            print(f"❌ 参数持久化失败：{str(e)}")
    
    def load_params(self) -> bool:
        """从持久化文件加载参数"""
        # 检查缓存目录是否存在
        if not os.path.exists(self.cache_dir):
            print("⚠️ 缓存目录不存在，无法加载参数")
            return False
        
        # 获取最新的缓存文件
        cache_files = [f for f in os.listdir(self.cache_dir) if f.startswith('usercf_swing_params_') and f.endswith('.pkl')]
        if not cache_files:
            print("⚠️ 未找到缓存文件，无法加载参数")
            return False
        
        # 按文件名排序，获取最新的缓存文件（文件名包含时间戳）
        cache_files.sort(reverse=True)
        latest_cache_file = os.path.join(self.cache_dir, cache_files[0])
        
        try:
            start_time = time.time()
            with open(latest_cache_file, 'rb') as f:
                params = pickle.load(f)
            
            # 恢复参数
            self.user_similarity = params.get('user_similarity')
            self.item_to_users = params.get('item_to_users')
            
            print(f"📥 成功加载缓存参数（耗时：{time.time() - start_time:.2f}秒）")
            print(f"📥 缓存文件：{latest_cache_file}")
            return True
        except Exception as e:
            print(f"❌ 加载缓存参数失败：{str(e)}")
            return False

# ----------------------
# 3. 主函数
# ----------------------
def main():
    # 1. 配置路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    
    # 2. 检查数据是否存在，不存在则生成
    data_files_exist = all([
        os.path.exists(os.path.join(data_dir, f)) 
        for f in ['user_table.csv', 'item_table.csv', 'interaction_table.csv']
    ])
    
    if not data_files_exist:
        print("📊 数据文件不存在，开始生成示例数据...")
        os.makedirs(data_dir, exist_ok=True)
        generator = DataGenerator()
        generator.generate_and_save_data(data_dir)
        print("✅ 数据生成完成！")
    
    # 3. 运行推荐器
    try:
        # 初始化推荐器，尝试从缓存加载参数
        recommender = UserCFSwingRecommender(data_dir, load_from_cache=True)
        
        # 确保加载了交互数据（即使缓存加载成功）
        if recommender.interaction_df is None:
            interaction_path = os.path.join(data_dir, 'interaction_table.csv')
            recommender.interaction_df = pd.read_csv(interaction_path)
        
        # 如果缓存加载失败或参数不完整，则重新计算
        if recommender.user_similarity is None:
            print("🔄 重新计算用户相似度...")
            recommender.calculate_user_similarity(use_weights=True)  # 使用交互权重
            # 计算完成后保存参数到缓存
            recommender.save_params()
        
        # 选择一个示例用户（从1000+的用户ID中选一个）
        sample_user_id = 1005
        
        # 检查用户是否有交互历史，如果没有则选择一个有交互历史的用户
        if recommender.interaction_df is not None:
            if recommender.interaction_df[recommender.interaction_df['user_id'] == sample_user_id].empty:
                print(f"⚠️ 用户{sample_user_id}无交互历史，切换到有交互历史的用户")
                # 获取有交互历史的用户ID列表
                active_users = recommender.interaction_df['user_id'].unique()
                if len(active_users) > 0:
                    sample_user_id = active_users[0]  # 选择第一个有交互历史的用户
                    print(f"🔄 已切换到用户{sample_user_id}")
        
        # 打印相似用户
        recommender.print_similar_users(sample_user_id, top_k=3)
        
        # 打印推荐结果
        recommender.print_recommendations(sample_user_id, top_n=5)
        
        # 评估推荐效果
        if recommender.interaction_df is not None:
            eval_results = recommender.evaluate(top_n=5)
            print(f"\n📊 推荐系统评估结果（Top5推荐）：")
            for metric, value in eval_results.items():
                print(f"  - {metric}：{value}")
    
    except Exception as e:
        print(f"\n❌ 程序运行失败：{str(e)}")

if __name__ == "__main__":
    main()