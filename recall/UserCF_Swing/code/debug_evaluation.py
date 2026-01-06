"""调试评估函数，找出准确度为0的原因"""
import pandas as pd
import sys
sys.path.append('.')
from UserCF_Swing import UserCFSwingRecommender
import os

# 初始化推荐器
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')

recommender = UserCFSwingRecommender(data_dir, load_from_cache=True)
if recommender.interaction_df is None:
    interaction_path = os.path.join(data_dir, 'interaction_table.csv')
    recommender.interaction_df = pd.read_csv(interaction_path)

# 选择一个测试用户
test_user_id = 1005
user_interactions = recommender.interaction_df[recommender.interaction_df['user_id'] == test_user_id]
print(f"\n用户 {test_user_id} 的交互情况:")
print(f"总交互数: {len(user_interactions)}")
print(f"交互物品: {sorted(user_interactions['item_id'].tolist())}")

# 模拟评估过程
test_ratio = 0.2
test_size = max(1, int(len(user_interactions) * test_ratio))
test_interactions = user_interactions.tail(test_size)
train_interactions = user_interactions.head(len(user_interactions) - test_size)

print(f"\n训练集交互数: {len(train_interactions)}")
print(f"训练集物品: {sorted(train_interactions['item_id'].tolist())}")
print(f"\n测试集交互数: {len(test_interactions)}")
print(f"测试集物品: {sorted(test_interactions['item_id'].tolist())}")

# 临时修改interaction_df
original_interaction_df = recommender.interaction_df.copy()
recommender.interaction_df = recommender.interaction_df[
    ~((recommender.interaction_df['user_id'] == test_user_id) & 
      (recommender.interaction_df['item_id'].isin(test_interactions['item_id'])))
]

# 获取推荐
recommended_items = recommender.recommend_items(test_user_id, top_n=5)
print(f"\n推荐结果数量: {len(recommended_items)}")
if recommended_items:
    print(f"推荐物品ID: {[item_id for item_id, _ in recommended_items]}")
    
    # 检查命中情况
    test_items = set(test_interactions['item_id'])
    recommended_item_ids = set([item_id for item_id, _ in recommended_items])
    hit_items = recommended_item_ids & test_items
    
    print(f"\n测试集物品: {test_items}")
    print(f"推荐物品: {recommended_item_ids}")
    print(f"命中物品: {hit_items}")
    print(f"召回率: {len(hit_items) / len(test_items) if test_items else 0:.4f}")
    print(f"精确率: {len(hit_items) / len(recommended_item_ids) if recommended_item_ids else 0:.4f}")
else:
    print("⚠️ 没有生成推荐！")

# 检查相似用户
similar_users = sorted(
    recommender.user_similarity.get(test_user_id, {}).items(),
    key=lambda x: x[1],
    reverse=True
)[:5]
print(f"\nTop 5 相似用户:")
for uid, sim in similar_users:
    similar_user_items = recommender.interaction_df[recommender.interaction_df['user_id'] == uid]
    print(f"  用户 {uid}: 相似度={sim:.4f}, 交互物品数={len(similar_user_items)}")

# 恢复
recommender.interaction_df = original_interaction_df.copy()

