import json
import pickle
import numpy as np
from scipy.sparse import load_npz, csr_matrix
import pandas as pd

def load_json(path):
    """加载JSON格式的映射文件（如user2idx.json）"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    """保存数据到JSON文件"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_pkl(path):
    """加载Pickle格式的集合文件（如C.pkl）"""
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pkl(data, path):
    """保存数据到Pickle文件"""
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_sparse_matrix(path):
    """加载npz格式的稀疏矩阵（如R.npz）"""
    return load_npz(path)

def save_sparse_matrix(matrix, path):
    """保存稀疏矩阵到npz文件"""
    np.savez(path, data=matrix.data, indices=matrix.indices, 
             indptr=matrix.indptr, shape=matrix.shape)

def build_mappings(ratings_df):
    """构建用户和物品的索引映射"""
    unique_users = sorted(ratings_df["user_id"].unique())
    unique_items = sorted(ratings_df["item_id"].unique())
    
    user2idx = {u: i for i, u in enumerate(unique_users)}
    item2idx = {v: j for j, v in enumerate(unique_items)}
    
    return user2idx, item2idx, len(unique_users), len(unique_items)

def build_matrices_and_sets(ratings_df, social_df, user2idx, item2idx, n_users, n_items):
    """构建评分矩阵R、社交矩阵T以及集合C, N, B"""
    # 映射数据
    ratings_df["user_idx"] = ratings_df["user_id"].map(user2idx)
    ratings_df["item_idx"] = ratings_df["item_id"].map(item2idx)
    
    # 构建用户-物品评分矩阵R (n_users × n_items)
    row = ratings_df["user_idx"].values
    col = ratings_df["item_idx"].values
    data = ratings_df["rating"].values
    R = csr_matrix((data, (row, col)), shape=(n_users, n_items))
    
    # 构建用户-用户社交矩阵T (n_users × n_users)
    social_df["user_idx"] = social_df["user_id"].map(user2idx)
    social_df["friend_idx"] = social_df["friend_id"].map(user2idx)
    
    # 过滤无效的社交关系（用户不在映射中的情况）
    valid_social = social_df.dropna(subset=["user_idx", "friend_idx"])
    row_social = valid_social["user_idx"].values.astype(int)
    col_social = valid_social["friend_idx"].values.astype(int)
    data_social = np.ones_like(row_social)
    T = csr_matrix((data_social, (row_social, col_social)), shape=(n_users, n_users))
    
    # 构建核心集合：C(i), N(i), B(j)
    C = [[] for _ in range(n_users)]  # C[i] = [物品j列表，用户i交互过的物品]
    N = [[] for _ in range(n_users)]  # N[i] = [用户o列表，用户i的社交好友]
    B = [[] for _ in range(n_items)]  # B[j] = [用户t列表，交互过物品j的用户]
    
    # 填充C(i)和B(j)
    for _, row in ratings_df.iterrows():
        u = int(row["user_idx"])
        v = int(row["item_idx"])
        C[u].append(v)
        B[v].append(u)
    
    # 填充N(i)
    for _, row in valid_social.iterrows():
        u = int(row["user_idx"])
        o = int(row["friend_idx"])
        N[u].append(o)
        N[o].append(u)  # 假设社交关系无向
    
    return R, T, C, N, B
