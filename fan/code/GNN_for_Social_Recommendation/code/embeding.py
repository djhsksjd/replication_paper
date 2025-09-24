import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

# 导入必要的库说明：
# pandas：用于数据读取和处理
# numpy：用于数值计算
# scipy.sparse.csr_matrix：用于构建稀疏矩阵

# 读取评分数据，文件格式为制表符分隔，包含用户ID、物品ID和评分
ratings = pd.read_csv("epinions/rating.mat", sep="\t", names=["user_id", "item_id", "rating"])
# 读取社交数据，文件格式为制表符分隔，包含用户ID和好友ID
social = pd.read_csv("epinions/trustnetwork.mat", sep="\t", names=["user_id", "friend_id"])

# 构建用户/物品索引映射（0-based），将原始ID映射为从0开始的连续索引
unique_users = sorted(ratings["user_id"].unique())
unique_items = sorted(ratings["item_id"].unique())
user2idx = {u: i for i, u in enumerate(unique_users)}
item2idx = {v: j for j, v in enumerate(unique_items)}
n_users = len(unique_users)  # 用户总数n
n_items = len(unique_items)  # 物品总数m

# 将原始数据中的用户ID和物品ID映射为新的索引
ratings["user_idx"] = ratings["user_id"].map(user2idx)
ratings["item_idx"] = ratings["item_id"].map(item2idx)
social["user_idx"] = social["user_id"].map(user2idx)
social["friend_idx"] = social["friend_id"].map(user2idx)

# 1. 构建用户-物品评分矩阵R (n_users × n_items)
# 获取用户索引、物品索引和评分数据
row = ratings["user_idx"].values
col = ratings["item_idx"].values
data = ratings["rating"].values
# 使用稀疏矩阵存储用户-物品评分数据，节省内存
R = csr_matrix((data, (row, col)), shape=(n_users, n_items))

# 2. 构建用户-用户社交矩阵T (n_users × n_users)
# 获取用户索引和好友索引数据
row_social = social["user_idx"].values
col_social = social["friend_idx"].values
# 社交关系数据初始化为1，表示存在连接
data_social = np.ones_like(row_social)
# 使用稀疏矩阵存储用户-用户社交关系数据
T = csr_matrix((data_social, (row_social, col_social)), shape=(n_users, n_users))

# 3. 构建核心集合：C(i), N(i), B(j)
C = [[] for _ in range(n_users)]  # C[i] = [物品j列表，用户i交互过的物品]
N = [[] for _ in range(n_users)]  # N[i] = [用户o列表，用户i的社交好友]
B = [[] for _ in range(n_items)]  # B[j] = [用户t列表，交互过物品j的用户]

# 填充C(i)和B(j)
# 遍历评分数据，记录每个用户交互过的物品和每个物品被哪些用户交互过
for _, row in ratings.iterrows():
    u = row["user_idx"]
    v = row["item_idx"]
    C[u].append(v)
    B[v].append(u)

# 填充N(i)
# 遍历社交数据，记录每个用户的社交好友
for _, row in social.iterrows():
    u = row["user_idx"]
    o = row["friend_idx"]
    N[u].append(o)
    N[o].append(u)  # 假设社交关系无向，若有向需删除此行（论文中T为无向图）

from sklearn.model_selection import train_test_split

# 仅对有评分的样本（O集合）划分
observed_ratings = ratings[ratings["rating"] > 0]  # O = {(u,i) | r_ij≠0}
# 将有评分的数据按80%比例划分为训练集和临时集
train_ratings, temp_ratings = train_test_split(observed_ratings, test_size=0.2, random_state=42)  # x=80%训练
# 将临时集按50%比例划分为验证集和测试集，最终形成10%验证，10%测试
val_ratings, test_ratings = train_test_split(temp_ratings, test_size=0.5, random_state=42)  # 10%验证，10%测试
