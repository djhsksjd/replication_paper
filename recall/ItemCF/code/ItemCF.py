import pandas as pd
import numpy as np
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import time
import pickle 
from datetime import datetime, timedelta


# ----------------------
# 1. 恢复数据生成器（解决数据缺失问题）
# ----------------------
class DataGenerator:
    def __init__(self, data_dir: str, num_users: int = 50, num_items: int = 30, 
                 interactions_per_user: tuple = (3, 10), num_categories: int = 5):
        self.data_dir = data_dir
        self.num_users = num_users
        self.num_items = num_items
        self.interactions_per_user = interactions_per_user
        self.num_categories = num_categories
        
        os.makedirs(data_dir, exist_ok=True)
        self.user_ids = [1000 + i for i in range(num_users)]
        self.item_ids = [2000 + i for i in range(num_items)]
        
        self.categories = ["电子产品", "家居用品", "服装配饰", "食品饮料", "图书文具"]
        self.item_type_templates = {
            "电子产品": ["智能手表", "无线耳机", "平板电脑", "蓝牙音箱", "移动电源"],
            "家居用品": ["保温杯", "抱枕", "收纳盒", "台灯", "香薰蜡烛"],
            "服装配饰": ["围巾", "帽子", "墨镜", "手链", "钱包"],
            "食品饮料": ["巧克力", "茶叶", "咖啡", "坚果礼盒", "果干"],
            "图书文具": ["笔记本", "钢笔", "小说", "工具书", "创意文具"]
        }
        
        self.interaction_types = ["点击", "收藏", "加入购物车", "购买"]
        self.start_date = datetime(2025, 9, 1)
        self.end_date = datetime(2025, 9, 30)
    
    def generate_user_data(self) -> pd.DataFrame:
        users = pd.DataFrame({
            'user_id': self.user_ids,
            'age': np.random.randint(18, 65, size=self.num_users),
            'gender': np.random.choice(['男', '女'], size=self.num_users, p=[0.55, 0.45])
        })
        return users
    
    def generate_item_data(self) -> pd.DataFrame:
        item_categories = np.random.choice(self.categories[:self.num_categories], size=self.num_items)
        item_names = []
        for category in item_categories:
            item_type = random.choice(self.item_type_templates[category])
            adjectives = ["高级", "智能", "时尚", "经典", "迷你", "便携", "多功能", "高品质"]
            adj = random.choice(adjectives)
            item_names.append(f"{adj}{item_type}")
        
        items = pd.DataFrame({
            'item_id': self.item_ids,
            'item_name': item_names,
            'category': item_categories,
            'price': np.round(np.random.uniform(10, 1000, size=self.num_items), 2)
        })
        return items
    
    def generate_interaction_data(self, users: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
        interactions = []
        for user_id in self.user_ids:
            num_interactions = random.randint(self.interactions_per_user[0], self.interactions_per_user[1])
            user_gender = users.loc[users['user_id'] == user_id, 'gender'].iloc[0]
            category_weights = np.ones(len(self.categories))
            
            if user_gender == '男':
                category_weights[self.categories.index("电子产品")] *= 1.5
                category_weights[self.categories.index("图书文具")] *= 1.2
            else:
                category_weights[self.categories.index("服装配饰")] *= 1.5
                category_weights[self.categories.index("家居用品")] *= 1.2
            
            category_weights = category_weights / category_weights.sum()
            interacted_items = set()
            
            while len(interacted_items) < num_interactions:
                preferred_category = np.random.choice(self.categories[:self.num_categories], p=category_weights)
                category_items = items[items['category'] == preferred_category]['item_id'].tolist()
                
                if category_items:
                    item_id = random.choice(category_items)
                    if item_id not in interacted_items:
                        interacted_items.add(item_id)
                        interaction_type = random.choices(
                            self.interaction_types,
                            weights=[0.4, 0.2, 0.2, 0.2],
                            k=1
                        )[0]
                        delta_days = random.randint(0, (self.end_date - self.start_date).days)
                        interaction_time = (self.start_date + timedelta(days=delta_days)).strftime('%Y-%m-%d')
                        interactions.append({
                            'user_id': user_id,
                            'item_id': item_id,
                            'interaction_type': interaction_type,
                            'interaction_time': interaction_time
                        })
        
        return pd.DataFrame(interactions)
    
    def generate_and_save_all(self) -> None:
        print(f"\n📊 开始生成数据（{self.num_users}用户 + {self.num_items}物品）")
        users = self.generate_user_data()
        items = self.generate_item_data()
        interactions = self.generate_interaction_data(users, items)
        
        user_path = os.path.join(self.data_dir, 'user_table.csv')
        item_path = os.path.join(self.data_dir, 'item_table.csv')
        interaction_path = os.path.join(self.data_dir, 'interaction_table.csv')
        
        users.to_csv(user_path, index=False)
        items.to_csv(item_path, index=False)
        interactions.to_csv(interaction_path, index=False)
        print(f"✅ 数据生成完成（保存路径：{self.data_dir}）\n")


# ----------------------
# 2. 修正后的ItemCF推荐器
# ----------------------
class ItemCFRecommender:
    # 修正1：补充cache_dir和load_from_cache参数，设置默认值
    def __init__(self, data_dir: str, cache_dir: Optional[str] = None, load_from_cache: bool = False):
        self.data_dir = data_dir
        self.user_df = None
        self.item_df = None
        self.interaction_df = None
        self.user_item_matrix = None
        self.item_similarity = None
        self.item_id_to_name = None
        
        # 修正2：正确初始化缓存目录（先赋值，再创建目录）
        self.cache_dir = cache_dir if cache_dir else os.path.join(data_dir, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)  # 确保缓存目录存在
        
        # 修正3：调整顺序：先加载数据，再加载缓存（缓存覆盖数据中的参数）
        self._load_data()
        self.load_from_cache = load_from_cache
        if self.load_from_cache:
            self.load_params()  # 加载缓存（覆盖item_id_to_name等）

    def save_params(self) -> None:
        """持久化保存关键参数"""
        start_time = time.time()
        params_to_save = {
            'user_item_matrix': self.user_item_matrix,
            'item_similarity': self.item_similarity,
            'item_id_to_name': self.item_id_to_name
        }
        
        # 生成带时间戳的缓存文件（便于区分版本）
        cache_file = os.path.join(
            self.cache_dir, 
            f'itemcf_params_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        )
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(params_to_save, f)  # pickle支持defaultdict类型
            print(f"💾 参数保存完成（耗时：{time.time() - start_time:.2f}秒）")
            print(f"💾 缓存路径：{cache_file}")
        except Exception as e:
            print(f"❌ 参数保存失败：{str(e)}")
    
    def load_params(self) -> bool:
        """加载最新缓存参数，返回是否成功"""
        # 筛选缓存文件（前缀+后缀匹配）
        cache_files = [
            f for f in os.listdir(self.cache_dir) 
            if f.startswith('itemcf_params_') and f.endswith('.pkl')
        ]
        if not cache_files:
            print("⚠️ 未找到缓存文件")
            return False
        
        # 按时间戳排序，取最新文件（文件名后缀是时间，逆序排）
        cache_files.sort(reverse=True)
        latest_cache = os.path.join(self.cache_dir, cache_files[0])
        
        try:
            start_time = time.time()
            with open(latest_cache, 'rb') as f:
                params = pickle.load(f)
            
            # 恢复参数（只恢复存在的键，避免KeyError）
            self.user_item_matrix = params.get('user_item_matrix')
            self.item_similarity = params.get('item_similarity')
            self.item_id_to_name = params.get('item_id_to_name')
            
            print(f"📥 缓存加载完成（耗时：{time.time() - start_time:.2f}秒）")
            print(f"📥 加载文件：{latest_cache}")
            return True
        except Exception as e:
            print(f"❌ 缓存加载失败：{str(e)}")
            return False

    def _load_data(self) -> None:
        """加载用户/物品/交互数据，缺失则报错"""
        start_time = time.time()
        file_paths = {
            'user': os.path.join(self.data_dir, 'user_table.csv'),
            'item': os.path.join(self.data_dir, 'item_table.csv'),
            'interaction': os.path.join(self.data_dir, 'interaction_table.csv')
        }
        
        # 检查缺失文件
        missing = [k for k, v in file_paths.items() if not os.path.exists(v)]
        if missing:
            raise FileNotFoundError(
                f"❌ 缺失{', '.join(missing)}数据文件！\n请先运行DataGenerator生成数据。"
            )
        
        
        try:
            self.user_df = pd.read_csv(file_paths['user'])
            self.item_df = pd.read_csv(file_paths['item'])
            self.interaction_df = pd.read_csv(file_paths['interaction'])
            # 从数据中构建item_id_to_name（若未加载缓存，用这个）
            self.item_id_to_name = dict(zip(self.item_df['item_id'], self.item_df['item_name']))
            

            self.interaction_df["weight"] = self.interaction_df["interaction_type"].apply(self._calculate_interaction_weight)

            print(f"📥 数据加载完成（耗时：{time.time() - start_time:.2f}秒）")
            print(f"📥 数据规模：{len(self.user_df)}用户 | {len(self.item_df)}物品 | {len(self.interaction_df)}交互")
        except Exception as e:
            raise RuntimeError(f"❌ 数据加载失败：{str(e)}") from e
    
    def _calculate_interaction_weight(self, interaction_type: str) -> int:
        """计算交互权重（覆盖所有4种交互类型）"""
        weight_map = {"点击": 1, "收藏": 2, "加入购物车": 3, "购买": 5}
        return weight_map.get(interaction_type, 0)
    
    def build_user_item_matrix(self) -> None:
        """构建用户-物品交互矩阵（带权重）"""
        if self.interaction_df is None:
            raise ValueError("❌ 未加载数据，请先调用_load_data()")
        
        start_time = time.time()
        # 计算交互权重
        self.interaction_df["weight"] = self.interaction_df["interaction_type"].apply(
            self._calculate_interaction_weight
        )
        
        # 构建矩阵（重复交互自动求和，空值填0）
        self.user_item_matrix = self.interaction_df[["user_id", "item_id", "weight"]].pivot_table(
            index="user_id", columns="item_id", values="weight", aggfunc="sum"
        ).fillna(0)
        
        print(f"\n🔧 矩阵构建完成（耗时：{time.time() - start_time:.2f}秒）")
        print(f"🔧 矩阵形状：{self.user_item_matrix.shape}（行=用户，列=物品）")
    
    def calculate_item_similarity(self) -> None:
        """计算物品余弦相似度（避免除零错误）"""
        if self.user_item_matrix is None:
            raise ValueError("❌ 未构建矩阵，请先调用build_user_item_matrix()")
        
        start_time = time.time()
        items = self.user_item_matrix.columns.tolist()
        item_vectors = self.user_item_matrix.values.T  # 转置为：物品×用户
        
        # 计算余弦相似度
        norms = np.linalg.norm(item_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10  # 避免无交互物品的模长为0
        normalized_vecs = item_vectors / norms
        sim_matrix = np.dot(normalized_vecs, normalized_vecs.T)
        
        # 转换为defaultdict（便于查询）
        self.item_similarity = defaultdict(dict)
        for i, item_i in enumerate(items):
            for j, item_j in enumerate(items):
                if i <= j:  # 避免重复计算（i-j与j-i相似度相同）
                    sim = round(sim_matrix[i, j], 4)
                    self.item_similarity[item_i][item_j] = sim
                    self.item_similarity[item_j][item_i] = sim
        
        print(f"🔍 相似度计算完成（耗时：{time.time() - start_time:.2f}秒）")
        print(f"🔍 计算物品数：{len(items)}")
    
    def recommend_items(self, user_id: int, top_n: int = 2, filter_interacted: bool = True) -> List[Tuple[int, float]]:
        """为用户推荐Top-N物品（过滤已交互）"""
        if self.item_similarity is None:
            raise ValueError("❌ 未计算相似度，请先调用calculate_item_similarity()")
        
        # 获取用户交互历史
        user_interacts = self.interaction_df[self.interaction_df["user_id"] == user_id]
        user_items = dict(zip(user_interacts["item_id"], user_interacts["weight"]))
        if not user_items:
            print(f"⚠️ 用户{user_id}无交互历史")
            return []
        
        # 计算推荐分数（相似度×交互权重，累加）
        candidate_scores = defaultdict(float)
        for item_i, weight_i in user_items.items():
            for item_j, sim in self.item_similarity.get(item_i, {}).items():
                if filter_interacted and item_j in user_items:
                    continue  # 过滤已交互物品
                candidate_scores[item_j] += sim * weight_i
        
        # 按分数降序取Top-N
        return sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def evaluate(self, test_user_ids: Optional[List[int]] = None, top_n: int = 2) -> Dict[str, float]:
        """评估推荐效果（召回率/精确率），修复边界错误"""
        if self.item_similarity is None:
            raise ValueError("❌ 未计算相似度，无法评估")
        
        # 获取有交互的用户（避免空数组）
        active_users = self.interaction_df["user_id"].unique()
        if len(active_users) == 0:
            print("⚠️ 无交互数据，无法评估")
            return {"测试用户数": 0, "平均召回率": 0.0, "平均精确率": 0.0}
        
        # 选择测试用户（默认20%，最少1个）
        if test_user_ids is None:
            sample_size = max(1, min(int(len(active_users) * 0.2), len(active_users)))
            test_user_ids = random.sample(list(active_users), sample_size)
        
        total_recall = 0.0
        total_precision = 0.0
        valid_users = 0
        
        for user_id in test_user_ids:
            # 真实正样本：用户交互过的物品
            real_items = set(self.interaction_df[self.interaction_df["user_id"] == user_id]["item_id"])
            if len(real_items) < 2:
                continue  # 交互太少，跳过
            
            # 推荐物品：模型输出
            recommended = set([iid for iid, _ in self.recommend_items(user_id, top_n)])
            if not recommended:
                continue
            
            # 命中物品：推荐与真实的交集
            hits = recommended & real_items
            valid_users += 1
            
            # 累加指标
            total_recall += len(hits) / len(real_items)
            total_precision += len(hits) / len(recommended)
        
        # 计算平均指标（避免除以0）
        avg_recall = round(total_recall / valid_users, 4) if valid_users else 0.0
        avg_precision = round(total_precision / valid_users, 4) if valid_users else 0.0
        
        return {
            "测试用户数": valid_users,
            "平均召回率": avg_recall,
            "平均精确率": avg_precision
        }
    
    def print_similarity_examples(self, sample_items: Optional[List[int]] = None) -> None:
        """打印物品相似度示例（随机选2个物品）"""
        if self.item_similarity is None:
            raise ValueError("❌ 未计算相似度，无法打印示例")
        
        # 随机选2个物品（避免传入不存在的ID）
        if sample_items is None:
            sample_items = random.sample(list(self.item_similarity.keys()), min(2, len(self.item_similarity)))
        
        print("\n📋 物品相似度示例（Top3相似物品）：")
        for item_i in sample_items:
            item_name_i = self.item_id_to_name.get(item_i, f"物品{item_i}")
            # 取除自身外的Top3相似物品
            similar_items = sorted(
                [(iid, sim) for iid, sim in self.item_similarity[item_i].items() if iid != item_i],
                key=lambda x: x[1], reverse=True
            )[:3]
            
            print(f"\n{item_name_i}（ID：{item_i}）的相似物品：")
            for item_j, sim in similar_items:
                item_name_j = self.item_id_to_name.get(item_j, f"物品{item_j}")
                print(f"  - {item_name_j}（ID：{item_j}）：相似度 {sim:.4f}")
    
    def print_recommendations(self, user_id: int, top_n: int = 2) -> None:
        """打印用户推荐结果（含交互历史）"""
        # 打印交互历史（前5条）
        user_interacts = self.interaction_df[self.interaction_df["user_id"] == user_id].head(5)
        print(f"\n👤 用户{user_id}的交互历史（前5条）：")
        if user_interacts.empty:
            print("  - 无交互历史")
        else:
            for _, row in user_interacts.iterrows():
                item_name = self.item_id_to_name.get(row['item_id'], f"物品{row['item_id']}")
                weight = self._calculate_interaction_weight(row['interaction_type'])
                print(f"  - {item_name}（ID：{row['item_id']}）：{row['interaction_type']}（权重：{weight}）")
        
        # 打印推荐结果
        recommendations = self.recommend_items(user_id, top_n)
        print(f"\n🎯 为用户{user_id}推荐Top{top_n}物品：")
        if not recommendations:
            print("  - 无推荐物品")
        else:
            for i, (item_id, score) in enumerate(recommendations, 1):
                item_name = self.item_id_to_name.get(item_id, f"物品{item_id}")
                # 安全获取物品类别（避免ID不存在）
                item_category = self.item_df[self.item_df["item_id"] == item_id]["category"].iloc[0] if \
                    not self.item_df[self.item_df["item_id"] == item_id].empty else "未知类别"
                print(f"  {i}. {item_name}（ID：{item_id}）")
                print(f"     - 推荐分数：{score:.4f} | 类别：{item_category}")


# ----------------------
# 3. 修正后的主函数（含数据生成）
# ----------------------
def main():
    # 1. 配置路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')  # 数据保存目录
    cache_dir = os.path.join(script_dir, 'itemcf_cache')  # 缓存保存目录（独立于数据）
    
    # 2. 检查数据是否存在，不存在则生成
    data_files = ['user_table.csv', 'item_table.csv', 'interaction_table.csv']
    data_exists = all([os.path.exists(os.path.join(data_dir, f)) for f in data_files])
    
    if not data_exists:
        print("⚠️ 未发现数据文件，开始生成...")
        generator = DataGenerator(
            data_dir=data_dir,
            num_users=50,
            num_items=30,
            interactions_per_user=(3, 10)
        )
        generator.generate_and_save_all()  # 生成数据
    
    # 3. 初始化推荐器（尝试加载缓存）
    try:
        recommender = ItemCFRecommender(
            data_dir=data_dir,
            cache_dir=cache_dir,
            load_from_cache=True  # 优先加载缓存
        )
        
        # 4. 若缓存缺失关键参数，重新计算
        need_recalculate = False
        if recommender.user_item_matrix is None:
            print("\n🔄 缓存中无用户-物品矩阵，重新构建...")
            recommender.build_user_item_matrix()
            need_recalculate = True
        
        if recommender.item_similarity is None:
            print("🔄 缓存中无物品相似度，重新计算...")
            recommender.calculate_item_similarity()
            need_recalculate = True
        
        # 5. 若重新计算，保存新参数到缓存
        if need_recalculate:
            recommender.save_params()
        
        # 6. 打印结果与评估
        recommender.print_similarity_examples()  # 相似度示例
        recommender.print_recommendations(user_id=1003, top_n=3)  # 推荐结果
        eval_res = recommender.evaluate(top_n=3)  # 评估
        
        # 打印评估结果
        print(f"\n📊 推荐系统评估结果（Top3）：")
        for metric, val in eval_res.items():
            print(f"  - {metric}：{val}")
    
    except Exception as e:
        print(f"\n❌ 程序运行失败：{str(e)}")


if __name__ == "__main__":
    main()