import pandas as pd
import numpy as np
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime, timedelta

# ----------------------
# 2. ItemCF推荐器（关键优化：适配数据生成器）
# ----------------------
class ItemCFRecommender:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.user_df = None
        self.item_df = None
        self.interaction_df = None
        self.user_item_matrix = None
        self.item_similarity = None
        self.item_id_to_name = None
        self._load_data()
    
    def _load_data(self) -> None:
        start_time = time.time()
        user_path = os.path.join(self.data_dir, 'user_table.csv')
        item_path = os.path.join(self.data_dir, 'item_table.csv')
        interaction_path = os.path.join(self.data_dir, 'interaction_table.csv')
        
        # 检查数据文件是否存在（避免报错）
        missing_files = [f for f in [user_path, item_path, interaction_path] if not os.path.exists(f)]
        if missing_files:
            raise FileNotFoundError(f"❌ 缺失数据文件：{', '.join(missing_files)}\n请先运行数据生成逻辑！")
        
        try:
            self.user_df = pd.read_csv(user_path)
            self.item_df = pd.read_csv(item_path)
            self.interaction_df = pd.read_csv(interaction_path)
            self.item_id_to_name = dict(zip(self.item_df['item_id'], self.item_df['item_name']))
            
            print(f"📥 数据加载完成（耗时：{time.time() - start_time:.2f}秒）")
            print(f"📥 数据规模：{len(self.user_df)}用户 | {len(self.item_df)}物品 | {len(self.interaction_df)}交互记录")
        except Exception as e:
            raise RuntimeError(f"❌ 数据加载失败：{str(e)}") from e
    
    def _calculate_interaction_weight(self, interaction_type: str) -> int:
        # 关键优化：补充“收藏”“加入购物车”的权重（原代码只处理了购买/点击，导致这两类权重为0）
        weight_map = {
            "点击": 1,
            "收藏": 2,
            "加入购物车": 3,
            "购买": 5  # 购买权重最高，符合实际业务逻辑
        }
        return weight_map.get(interaction_type, 0)
    
    def build_user_item_matrix(self) -> None:
        if self.interaction_df is None:
            raise ValueError("❌ 未加载数据，请先调用_load_data()")
        
        start_time = time.time()
        # 计算交互权重（现在支持所有4种交互类型）
        self.interaction_df["weight"] = self.interaction_df["interaction_type"].apply(self._calculate_interaction_weight)
        
        # 构建矩阵（用pivot_table，重复交互自动求和）
        self.user_item_matrix = self.interaction_df[["user_id", "item_id", "weight"]].pivot_table(
            index="user_id", columns="item_id", values="weight", aggfunc="sum"
        ).fillna(0)
        
        print(f"\n🔧 用户-物品矩阵构建完成（耗时：{time.time() - start_time:.2f}秒）")
        print(f"🔧 矩阵形状：{self.user_item_matrix.shape}（行=用户，列=物品）")
    
    def calculate_item_similarity(self) -> None:
        if self.user_item_matrix is None:
            raise ValueError("❌ 未构建用户-物品矩阵，请先调用build_user_item_matrix()")
        
        start_time = time.time()
        items = self.user_item_matrix.columns.tolist()
        item_vectors = self.user_item_matrix.values.T  # 转置为“物品-用户”向量
        
        # 余弦相似度计算（避免除零错误）
        norms = np.linalg.norm(item_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10  # 防止无交互物品的模长为0导致除以0
        normalized_vectors = item_vectors / norms
        similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
        
        # 转换为字典（便于查询）
        self.item_similarity = defaultdict(dict)
        for i, item_i in enumerate(items):
            for j, item_j in enumerate(items):
                if i <= j:  # 避免重复计算（i-j和j-i相似度相同）
                    sim = round(similarity_matrix[i, j], 4)
                    self.item_similarity[item_i][item_j] = sim
                    self.item_similarity[item_j][item_i] = sim
        
        print(f"🔍 物品相似度计算完成（耗时：{time.time() - start_time:.2f}秒）")
        print(f"🔍 共计算 {len(items)} 个物品的相似度矩阵")
    
    def recommend_items(self, user_id: int, top_n: int = 2, filter_interacted: bool = True) -> List[Tuple[int, float]]:
        if self.item_similarity is None:
            raise ValueError("❌ 未计算物品相似度，请先调用calculate_item_similarity()")
        
        # 1. 获取用户交互过的物品及权重
        user_interactions = self.interaction_df[self.interaction_df["user_id"] == user_id]
        user_items = dict(zip(user_interactions["item_id"], user_interactions["weight"]))
        if not user_items:
            print(f"⚠️ 用户{user_id}无交互历史，无法推荐")
            return []
        
        # 2. 计算候选物品分数（相似度 × 原物品权重，累加）
        candidate_scores = defaultdict(float)
        for item_i, weight_i in user_items.items():
            for item_j, sim in self.item_similarity.get(item_i, {}).items():
                if filter_interacted and item_j in user_items:
                    continue  # 过滤已交互物品，避免重复推荐
                candidate_scores[item_j] += sim * weight_i
        
        # 3. 按分数排序，取Top-N
        return sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def evaluate(self, test_user_ids: Optional[List[int]] = None, top_n: int = 2) -> Dict[str, float]:
        # 优化评估逻辑：基于“交互历史的相似物品”作为伪正样本（更贴近实际评估）
        if self.item_similarity is None:
            raise ValueError("❌ 未计算物品相似度，无法评估")
        
        # 选择测试用户（默认选20%有交互的用户）
        active_users = self.interaction_df["user_id"].unique()
        if test_user_ids is None:
            test_user_ids = random.sample(list(active_users), max(1, int(len(active_users) * 0.2)))
        
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
            
            # 命中物品：推荐中包含的真实正样本（此处简化为“相似物品”视为潜在正样本）
            hit_items = recommended_items & real_items
            valid_users += 1
            
            # 计算召回率（命中/真实总数）和精确率（命中/推荐总数）
            total_recall += len(hit_items) / len(real_items) if real_items else 0.0
            total_precision += len(hit_items) / len(recommended_items) if recommended_items else 0.0
        
        # 返回平均指标
        return {
            "测试用户数": valid_users,
            "平均召回率": round(total_recall / valid_users, 4) if valid_users else 0.0,
            "平均精确率": round(total_precision / valid_users, 4) if valid_users else 0.0
        }
    
    def print_similarity_examples(self, sample_items: Optional[List[int]] = None) -> None:
        # 关键优化：默认取数据生成器的物品ID（2000+），而非原代码的201/202（不存在）
        if sample_items is None:
            sample_items = random.sample(list(self.item_similarity.keys()), 2)  # 随机选2个物品
        
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
        # 1. 打印用户交互历史（前5条，避免过长）
        user_interactions = self.interaction_df[self.interaction_df["user_id"] == user_id].head(5)
        print(f"\n👤 用户{user_id}的交互历史（前5条）：")
        if user_interactions.empty:
            print("  - 无交互历史")
        else:
            for _, row in user_interactions.iterrows():
                item_name = self.item_id_to_name.get(row['item_id'], f"物品{row['item_id']}")
                print(f"  - {item_name}（ID：{row['item_id']}）：{row['interaction_type']}（权重：{self._calculate_interaction_weight(row['interaction_type'])}）")
        
        # 2. 打印推荐结果
        recommendations = self.recommend_items(user_id, top_n)
        print(f"\n🎯 为用户{user_id}推荐的Top{top_n}物品：")
        if not recommendations:
            print("  - 无推荐物品")
        else:
            for i, (item_id, score) in enumerate(recommendations, 1):
                item_name = self.item_id_to_name.get(item_id, f"物品{item_id}")
                item_category = self.item_df[self.item_df["item_id"] == item_id]["category"].iloc[0]
                print(f"  {i}. {item_name}（ID：{item_id}）")
                print(f"     - 推荐分数：{score:.4f} | 类别：{item_category}")


# ----------------------
# 3. 主函数（整合“数据生成+推荐运行”）
# ----------------------
def main():
    # 1. 配置路径（脚本所在目录下的data文件夹）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    
    # 2. 检查数据是否存在，不存在则生成
    data_files_exist = all([
        os.path.exists(os.path.join(data_dir, f)) 
        for f in ['user_table.csv', 'item_table.csv', 'interaction_table.csv']
    ])
    
    if not data_files_exist:
        print("⚠️ 未发现数据文件，开始生成...")
        generator = DataGenerator(
            data_dir=data_dir,
            num_users=50,      # 可调整：用户数量
            num_items=30,      # 可调整：物品数量
            interactions_per_user=(3, 10)  # 可调整：每个用户的交互次数范围
        )
        generator.generate_and_save_all()
    else:
        print(f"✅ 数据文件已存在（路径：{data_dir}）\n")
    
    # 3. 运行推荐器
    try:
        recommender = ItemCFRecommender(data_dir)
        recommender.build_user_item_matrix()
        recommender.calculate_item_similarity()
        
        # 打印相似度示例
        recommender.print_similarity_examples()
        
        # 为用户1003推荐（数据生成器用户ID是1000+，1003一定存在）
        target_user = 1003
        recommender.print_recommendations(target_user, top_n=3)  # 推荐Top3，更直观
        
        # 评估推荐效果
        eval_results = recommender.evaluate(top_n=3)
        print(f"\n📊 推荐系统评估结果（Top3推荐）：")
        for metric, value in eval_results.items():
            print(f"  - {metric}：{value}")
    
    except Exception as e:
        print(f"\n❌ 程序运行失败：{str(e)}")


if __name__ == "__main__":
    main()