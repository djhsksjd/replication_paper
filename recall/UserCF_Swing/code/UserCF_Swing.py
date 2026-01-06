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
import copy


# ----------------------
# 1. 数据生成器类
# ----------------------
class DataGenerator:
    def __init__(self):
        # 设置数据规模参数
        self.num_users = 500        # 用户数量
        self.num_items = 1000       # 物品数量
        self.num_interactions = 15000 # 交互记录数量（增加到15000，平均每用户30个交互）
        
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
    
    def calculate_user_similarity(self, use_weights: bool = False, return_steps: bool = False) -> Optional[Dict]:
        """使用Swing算法计算用户相似度
        
        Args:
            use_weights: 是否使用交互权重（购买>收藏>点击）
            return_steps: 是否返回详细计算步骤
        """
        if self.item_to_users is None:
            raise ValueError("❌ 未构建物品到用户倒排表")
        
        start_time = time.time()
        self.user_similarity = defaultdict(dict)
        
        # 用于存储计算步骤
        calculation_steps = []
        example_steps = []  # 存储示例计算步骤（前3个）
        example_count = 0
        
        # 遍历每个物品的用户列表
        total_items = len(self.item_to_users)
        processed_items = 0
        
        for item_id, users in self.item_to_users.items():
            processed_items += 1
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
                    step_detail = {
                        'item_id': item_id,
                        'user_u': u,
                        'user_v': v,
                        'user_count': user_count,
                        'penalty': round(penalty, 6),
                        'base_contribution': round(contribution, 6),
                        'weights': {}
                    }
                    
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
                            u_interaction_type = u_interaction['interaction_type'].iloc[0]
                            v_interaction_type = v_interaction['interaction_type'].iloc[0]
                            u_weight = self._get_interaction_weight(u_interaction_type)
                            v_weight = self._get_interaction_weight(v_interaction_type)
                            contribution *= (u_weight * v_weight)
                            
                            step_detail['weights'] = {
                                'user_u': {'type': u_interaction_type, 'weight': u_weight},
                                'user_v': {'type': v_interaction_type, 'weight': v_weight},
                                'combined': u_weight * v_weight
                            }
                    
                    step_detail['final_contribution'] = round(contribution, 6)
                    
                    # 记录示例步骤（前3个）
                    if example_count < 3:
                        example_steps.append(step_detail.copy())
                        example_count += 1
                    
                    # 累加到用户相似度矩阵
                    if v not in self.user_similarity[u]:
                        self.user_similarity[u][v] = 0.0
                    if u not in self.user_similarity[v]:
                        self.user_similarity[v][u] = 0.0
                    
                    old_sim_u_v = self.user_similarity[u][v]
                    old_sim_v_u = self.user_similarity[v][u]
                    
                    self.user_similarity[u][v] += contribution
                    self.user_similarity[v][u] += contribution
                    
                    step_detail['similarity_before'] = round(old_sim_u_v, 6)
                    step_detail['similarity_after'] = round(self.user_similarity[u][v], 6)
                    calculation_steps.append(step_detail)
        
        print(f"🔍 用户相似度计算完成（耗时：{time.time() - start_time:.2f}秒）")
        print(f"🔍 计算了 {len(self.user_similarity)} 个用户的相似度")
        print(f"🔍 处理了 {total_items} 个物品，共 {len(calculation_steps)} 个用户对")
        
        if return_steps:
            # 转换example_steps中的numpy类型
            converted_example_steps = []
            for step in example_steps:
                converted_step = {
                    'item_id': int(step['item_id']),
                    'user_u': int(step['user_u']),
                    'user_v': int(step['user_v']),
                    'user_count': int(step['user_count']),
                    'penalty': float(step['penalty']),
                    'base_contribution': float(step['base_contribution']),
                    'final_contribution': float(step['final_contribution']),
                    'similarity_before': float(step['similarity_before']),
                    'similarity_after': float(step['similarity_after']),
                    'weights': {}
                }
                if step.get('weights'):
                    converted_step['weights'] = {
                        'user_u': {
                            'type': str(step['weights']['user_u']['type']),
                            'weight': int(step['weights']['user_u']['weight'])
                        },
                        'user_v': {
                            'type': str(step['weights']['user_v']['type']),
                            'weight': int(step['weights']['user_v']['weight'])
                        },
                        'combined': int(step['weights']['combined'])
                    }
                converted_example_steps.append(converted_step)
            
            return {
                'total_items': int(total_items),
                'total_pairs': int(len(calculation_steps)),
                'num_users': int(len(self.user_similarity)),
                'example_steps': converted_example_steps,
                'time_cost': round(float(time.time() - start_time), 2)
            }
        
        return None
    
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
    
    def recommend_items_with_reasons_and_steps(self, user_id: int, top_n: int = 10, k_similar_users: int = 50) -> Tuple[List[Tuple[int, float, Dict]], Dict]:
        """为目标用户推荐物品，并返回推荐原因和详细计算步骤
        
        Returns:
            (recommendations, steps_dict): 推荐列表和计算步骤
        """
        if self.user_similarity is None:
            raise ValueError("❌ 未计算用户相似度")
        
        steps = {
            'user_id': user_id,
            'steps': []
        }
        
        # 获取用户已交互的物品集合
        user_items = set()
        if self.interaction_df is not None:
            user_items = set(self.interaction_df[self.interaction_df['user_id'] == user_id]['item_id'])
        
        steps['steps'].append({
            'step': 1,
            'description': f'获取用户 {user_id} 的交互历史',
            'user_items': [int(item) for item in list(user_items)[:10]],  # 只显示前10个，转换为int
            'total_items': int(len(user_items))
        })
        
        if not user_items:
            steps['steps'].append({
                'step': 2,
                'description': f'用户 {user_id} 无交互历史，无法推荐',
                'error': True
            })
            return [], steps
        
        # 获取目标用户的相似用户（按相似度排序）
        similar_users = sorted(
            self.user_similarity.get(user_id, {}).items(),
            key=lambda x: x[1],
            reverse=True
        )[:k_similar_users]
        
        steps['steps'].append({
            'step': 2,
            'description': f'找到 {int(len(similar_users))} 个相似用户',
            'similar_users': [
                {'user_id': int(uid), 'similarity': round(float(sim), 4)} 
                for uid, sim in similar_users[:5]  # 只显示前5个
            ]
        })
        
        if not similar_users:
            steps['steps'].append({
                'step': 3,
                'description': f'未找到用户 {user_id} 的相似用户',
                'error': True
            })
            return [], steps
        
        # 计算候选物品分数，并记录推荐原因
        item_scores = defaultdict(float)
        item_reasons = defaultdict(lambda: {'similar_users': []})
        item_calculation_steps = defaultdict(list)  # 记录每个物品的计算步骤
        
        # 获取用户交互过的物品（用于找共同物品）
        user_item_set = set(user_items)
        
        # 预先计算每个相似用户与目标用户的共同物品
        similar_user_common_items = {}
        for similar_user, similarity in similar_users:
            if self.interaction_df is not None:
                similar_user_items = self.interaction_df[self.interaction_df['user_id'] == similar_user]
                similar_user_item_set = set(similar_user_items['item_id'])
                common_items = list(user_item_set & similar_user_item_set)[:5]  # 最多5个共同物品
                similar_user_common_items[similar_user] = {
                    'common_items': common_items,
                    'similarity': similarity
                }
        
        steps['steps'].append({
            'step': 3,
            'description': '计算每个相似用户与目标用户的共同物品',
            'common_items_example': {
                'similar_user': int(similar_users[0][0]) if similar_users else None,
                'common_items': [int(item) for item in similar_user_common_items.get(similar_users[0][0], {}).get('common_items', [])[:3]] if similar_users else []
            }
        })
        
        # 记录计算过程
        calculation_details = []
        
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
                        score_contribution = similarity * weight
                        item_scores[item_id] += score_contribution
                        
                        # 记录计算步骤（只记录前几个物品的详细步骤）
                        if len(calculation_details) < 10:
                            calculation_details.append({
                                'item_id': int(item_id),
                                'similar_user': int(similar_user),
                                'similarity': round(float(similarity), 4),
                                'interaction_type': str(row['interaction_type']),
                                'weight': int(weight),
                                'contribution': round(float(score_contribution), 4),
                                'item_score_before': round(float(item_scores[item_id] - score_contribution), 4),
                                'item_score_after': round(float(item_scores[item_id]), 4)
                            })
                        
                        # 记录推荐原因：哪些相似用户推荐了这个物品（按贡献度排序）
                        if len(item_reasons[item_id]['similar_users']) < 5:  # 最多记录5个相似用户
                            contribution = similarity * weight
                            item_reasons[item_id]['similar_users'].append({
                                'user_id': similar_user,
                                'similarity': similarity,
                                'contribution': contribution,
                                'interaction_type': row['interaction_type'],
                                'weight': weight
                            })
                            # 按贡献度排序
                            item_reasons[item_id]['similar_users'].sort(key=lambda x: x['contribution'], reverse=True)
        
        steps['steps'].append({
            'step': 4,
            'description': f'计算候选物品分数，共 {int(len(item_scores))} 个候选物品',
            'calculation_example': calculation_details[:5]  # 显示前5个计算示例
        })
        
        # 按分数排序，返回Top-N物品及推荐原因
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        steps['steps'].append({
            'step': 5,
            'description': f'排序并选择 Top-{top_n} 推荐物品',
            'top_items': [
                {'item_id': int(item_id), 'score': round(float(score), 4)} 
                for item_id, score in sorted_items
            ]
        })
        
        result = []
        for item_id, score in sorted_items:
            reason = item_reasons.get(item_id, {'similar_users': []})
            # 为每个推荐物品添加共同物品信息（从Top相似用户获取）
            if reason['similar_users']:
                top_similar_user = reason['similar_users'][0]['user_id']
                if top_similar_user in similar_user_common_items:
                    reason['common_items'] = similar_user_common_items[top_similar_user]['common_items']
                else:
                    reason['common_items'] = []
            else:
                reason['common_items'] = []
            result.append((item_id, score, reason))
        
        return result, steps
    
    def recommend_items_with_reasons(self, user_id: int, top_n: int = 10, k_similar_users: int = 50) -> List[Tuple[int, float, Dict]]:
        """为目标用户推荐物品，并返回推荐原因
        
        Args:
            user_id: 目标用户ID
            top_n: 推荐物品数量
            k_similar_users: 参考的相似用户数量
            
        Returns:
            List[Tuple[item_id, score, reason_dict]]: 推荐物品列表，每个元素包含物品ID、分数和推荐原因
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
        
        # 计算候选物品分数，并记录推荐原因
        item_scores = defaultdict(float)
        item_reasons = defaultdict(lambda: {'similar_users': []})
        
        # 获取用户交互过的物品（用于找共同物品）
        user_item_set = set(user_items)
        
        # 预先计算每个相似用户与目标用户的共同物品
        similar_user_common_items = {}
        for similar_user, similarity in similar_users:
            if self.interaction_df is not None:
                similar_user_items = self.interaction_df[self.interaction_df['user_id'] == similar_user]
                similar_user_item_set = set(similar_user_items['item_id'])
                common_items = list(user_item_set & similar_user_item_set)[:5]  # 最多5个共同物品
                similar_user_common_items[similar_user] = {
                    'common_items': common_items,
                    'similarity': similarity
                }
        
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
                        
                        # 记录推荐原因：哪些相似用户推荐了这个物品（按贡献度排序）
                        if len(item_reasons[item_id]['similar_users']) < 5:  # 最多记录5个相似用户
                            contribution = similarity * weight
                            item_reasons[item_id]['similar_users'].append({
                                'user_id': similar_user,
                                'similarity': similarity,
                                'contribution': contribution,
                                'interaction_type': row['interaction_type'],
                                'weight': weight
                            })
                            # 按贡献度排序
                            item_reasons[item_id]['similar_users'].sort(key=lambda x: x['contribution'], reverse=True)
        
        # 按分数排序，返回Top-N物品及推荐原因
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        result = []
        for item_id, score in sorted_items:
            reason = item_reasons.get(item_id, {'similar_users': []})
            # 为每个推荐物品添加共同物品信息（从Top相似用户获取）
            if reason['similar_users']:
                top_similar_user = reason['similar_users'][0]['user_id']
                if top_similar_user in similar_user_common_items:
                    reason['common_items'] = similar_user_common_items[top_similar_user]['common_items']
                else:
                    reason['common_items'] = []
            else:
                reason['common_items'] = []
            result.append((item_id, score, reason))
        
        return result
    
    def evaluate(self, test_user_ids: Optional[List[int]] = None, top_n: int = 10, test_ratio: float = 0.2) -> Dict[str, float]:
        """评估推荐系统效果
        
        Args:
            test_user_ids: 测试用户ID列表，如果为None则随机选择
            top_n: 推荐物品数量
            test_ratio: 每个用户用于测试的交互比例（默认20%作为测试集）
        """
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
        no_recommendation_count = 0  # 统计无法生成推荐的用户数
        
        # 保存原始interaction_df，用于恢复
        original_interaction_df = self.interaction_df.copy()
        
        for user_id in test_user_ids:
            # 获取该用户的所有交互
            user_interactions = self.interaction_df[self.interaction_df["user_id"] == user_id].copy()
            if len(user_interactions) < 2:
                continue  # 交互太少，评估无意义
            
            # 按时间排序（如果有时间字段）或随机打乱
            if 'interaction_time' in user_interactions.columns:
                user_interactions = user_interactions.sort_values('interaction_time')
            
            # 分割训练集和测试集
            test_size = max(1, int(len(user_interactions) * test_ratio))
            test_interactions = user_interactions.tail(test_size)
            train_interactions = user_interactions.head(len(user_interactions) - test_size)
            
            # 如果训练集为空，跳过
            if len(train_interactions) == 0:
                continue
            
            # 临时修改interaction_df，只保留训练集交互（用于生成推荐）
            self.interaction_df = self.interaction_df[
                ~((self.interaction_df['user_id'] == user_id) & (self.interaction_df['item_id'].isin(test_interactions['item_id'])))
            ]
            
            # 真实正样本：测试集中的物品（用户未来会交互的物品）
            test_items = set(test_interactions['item_id'])
            
            # 推荐物品：基于训练集生成的推荐
            recommended_items = set([item_id for item_id, _ in self.recommend_items(user_id, top_n)])
            if not recommended_items:
                no_recommendation_count += 1
                # 恢复interaction_df
                self.interaction_df = original_interaction_df.copy()
                continue
            
            # 命中物品：推荐中包含的测试集物品
            hit_items = recommended_items & test_items
            valid_users += 1
            
            # 计算召回率和精确率
            total_recall += len(hit_items) / len(test_items) if test_items else 0.0
            total_precision += len(hit_items) / len(recommended_items) if recommended_items else 0.0
            
            # 恢复interaction_df
            self.interaction_df = original_interaction_df.copy()
        
        # 返回平均指标
        result = {
            "测试用户数": valid_users,
            "平均召回率": round(total_recall / valid_users, 4) if valid_users else 0.0,
            "平均精确率": round(total_precision / valid_users, 4) if valid_users else 0.0
        }
        
        # 添加调试信息
        if no_recommendation_count > 0:
            result["无法生成推荐的用户数"] = no_recommendation_count
        
        return result
    
    def evaluate_with_global_split(self, test_ratio: float = 0.2, top_n: int = 10, min_user_interactions: int = 5) -> Dict[str, float]:
        """使用全局训练/测试分割评估推荐系统效果（更准确的评估方法）
        
        Args:
            test_ratio: 测试集比例（默认20%）
            top_n: 推荐物品数量
            min_user_interactions: 用户最少交互数（低于此值的用户不参与评估）
        """
        if self.interaction_df is None:
            raise ValueError("❌ 未加载交互数据，无法评估")
        
        print(f"\n🔄 开始全局训练/测试分割评估（测试集比例：{test_ratio:.0%}）...")
        start_time = time.time()
        
        # 保存原始数据
        original_interaction_df = self.interaction_df.copy()
        original_user_similarity = copy.deepcopy(self.user_similarity) if self.user_similarity else None
        original_item_to_users = copy.deepcopy(self.item_to_users) if self.item_to_users else None
        
        # 全局分割：按时间排序后分割
        if 'interaction_time' in self.interaction_df.columns:
            self.interaction_df = self.interaction_df.sort_values('interaction_time')
        else:
            # 如果没有时间字段，随机打乱
            self.interaction_df = self.interaction_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 计算分割点
        split_idx = int(len(self.interaction_df) * (1 - test_ratio))
        train_df = self.interaction_df.iloc[:split_idx].copy()
        test_df = self.interaction_df.iloc[split_idx:].copy()
        
        print(f"📊 训练集：{len(train_df)}条交互，测试集：{len(test_df)}条交互")
        
        # 使用训练集重新构建相似度矩阵
        print("🔄 基于训练集重新计算用户相似度...")
        self.interaction_df = train_df
        self._build_item_to_users()
        self.calculate_user_similarity(use_weights=True)
        
        # 获取测试集中的用户（至少有min_user_interactions个交互）
        test_users = test_df['user_id'].value_counts()
        test_users = test_users[test_users >= min_user_interactions].index.tolist()
        
        print(f"📊 测试用户数：{len(test_users)}（至少{min_user_interactions}个交互）")
        
        # 评估每个测试用户
        total_recall = 0.0
        total_precision = 0.0
        total_hits = 0
        valid_users = 0
        no_recommendation_count = 0
        
        for user_id in test_users:
            # 该用户在测试集中的真实交互物品
            test_items = set(test_df[test_df['user_id'] == user_id]['item_id'])
            
            # 该用户在训练集中的交互物品（用于过滤推荐）
            train_items = set(train_df[train_df['user_id'] == user_id]['item_id'])
            
            # 如果训练集中没有交互，跳过（冷启动问题）
            if len(train_items) == 0:
                continue
            
            # 生成推荐（基于训练集）
            recommended_items = set([item_id for item_id, _ in self.recommend_items(user_id, top_n)])
            
            if not recommended_items:
                no_recommendation_count += 1
                continue
            
            # 计算命中
            hit_items = recommended_items & test_items
            valid_users += 1
            total_hits += len(hit_items)
            
            # 计算召回率和精确率
            total_recall += len(hit_items) / len(test_items) if test_items else 0.0
            total_precision += len(hit_items) / len(recommended_items) if recommended_items else 0.0
        
        # 计算平均指标
        avg_recall = round(total_recall / valid_users, 4) if valid_users > 0 else 0.0
        avg_precision = round(total_precision / valid_users, 4) if valid_users > 0 else 0.0
        
        result = {
            "测试用户数": valid_users,
            "平均召回率": avg_recall,
            "平均精确率": avg_precision,
            "总命中数": total_hits,
            "评估耗时": f"{time.time() - start_time:.2f}秒"
        }
        
        if no_recommendation_count > 0:
            result["无法生成推荐的用户数"] = no_recommendation_count
        
        print(f"✅ 评估完成（耗时：{time.time() - start_time:.2f}秒）")
        
        # 恢复原始数据（在返回前恢复）
        self.interaction_df = original_interaction_df
        self.user_similarity = original_user_similarity
        self.item_to_users = original_item_to_users
        
        return result
    
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
        
        # 2. 打印推荐结果（带推荐原因）
        recommendations = self.recommend_items_with_reasons(user_id, top_n)
        print(f"\n🎯 为用户{user_id}推荐的Top{top_n}物品：")
        if not recommendations:
            print("  - 无推荐物品")
        else:
            for i, (item_id, score, reason) in enumerate(recommendations, 1):
                item_info = self.item_df[self.item_df['item_id'] == item_id]
                item_name = item_info['item_name'].iloc[0] if not item_info.empty else f"物品{item_id}"
                item_category = item_info['category'].iloc[0] if not item_info.empty else '未知'
                item_price = item_info['price'].iloc[0] if not item_info.empty else '未知'
                
                print(f"\n  {i}. {item_name}（ID：{item_id}）")
                print(f"     📊 推荐分数：{score:.4f} | 类别：{item_category} | 价格：¥{item_price}")
                
                # 显示推荐原因
                if reason['similar_users']:
                    print(f"     💡 推荐原因：")
                    # 显示Top 3相似用户
                    top_similar_users = reason['similar_users'][:3]
                    for j, sim_user_info in enumerate(top_similar_users, 1):
                        sim_user_id = sim_user_info['user_id']
                        sim_similarity = sim_user_info['similarity']
                        sim_interaction = sim_user_info['interaction_type']
                        sim_contribution = sim_user_info['contribution']
                        
                        # 获取相似用户信息
                        sim_user_data = self.user_df[self.user_df['user_id'] == sim_user_id]
                        sim_age = sim_user_data['age'].iloc[0] if not sim_user_data.empty else '未知'
                        sim_gender = sim_user_data['gender'].iloc[0] if not sim_user_data.empty else '未知'
                        
                        print(f"        {j}. 用户{sim_user_id}（{sim_gender}，{sim_age}岁）")
                        print(f"           - 相似度：{sim_similarity:.4f} | 对该物品：{sim_interaction} | 贡献度：{sim_contribution:.4f}")
                    
                    # 显示共同交互的物品（解释为什么相似）
                    if reason.get('common_items'):
                        common_items = reason['common_items'][:3]  # 最多显示3个
                        if common_items:
                            common_item_names = []
                            for common_item_id in common_items:
                                common_item_info = self.item_df[self.item_df['item_id'] == common_item_id]
                                if not common_item_info.empty:
                                    common_item_name = common_item_info['item_name'].iloc[0]
                                    common_item_names.append(common_item_name)
                            
                            if common_item_names:
                                print(f"        🔗 与Top相似用户共同喜欢：{', '.join(common_item_names)}")
                else:
                    print(f"     💡 推荐原因：基于协同过滤算法推荐")
    
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
        
        # 评估推荐效果（使用改进的全局分割方法）
        if recommender.interaction_df is not None:
            print("\n" + "="*60)
            print("📊 推荐系统评估（全局训练/测试分割）")
            print("="*60)
            eval_results = recommender.evaluate_with_global_split(
                test_ratio=0.2,  # 20%作为测试集
                top_n=10,  # 推荐Top10物品（增加推荐数量提高命中率）
                min_user_interactions=5  # 至少5个交互的用户才参与评估
            )
            for metric, value in eval_results.items():
                print(f"  - {metric}：{value}")
            
            # 同时运行原来的评估方法作为对比
            print("\n" + "="*60)
            print("📊 推荐系统评估（按用户分割）")
            print("="*60)
            eval_results_old = recommender.evaluate(
                top_n=10,  # 增加推荐数量
                test_ratio=0.1  # 降低测试比例到10%，增加训练数据
            )
            for metric, value in eval_results_old.items():
                print(f"  - {metric}：{value}")
    
    except Exception as e:
        print(f"\n❌ 程序运行失败：{str(e)}")

if __name__ == "__main__":
    main()