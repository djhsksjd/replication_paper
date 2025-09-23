import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from config import config
from utils.data_utils import (
    build_mappings, build_matrices_and_sets,
    save_json, save_pkl, save_sparse_matrix
)
from utils.path_utils import create_directories

def preprocess_dataset(dataset_name="ciao"):
    """
    预处理数据集，生成模型所需的各种数据结构
    
    参数:
        dataset_name: 数据集名称，"ciao"或"epinions"
    """
    print(f"Starting preprocessing for {dataset_name} dataset...")
    
    # 1. 定义路径
    raw_data_path = config["root_path"] / "data" / "raw"
    processed_data_path = config["root_path"] / "data" / "processed" / dataset_name
    splits_path = config["root_path"] / "data" / "splits"
    
    # 创建目录
    create_directories([processed_data_path, splits_path])
    
    # 2. 读取原始数据
    try:
        ratings = pd.read_csv(
            raw_data_path / f"{dataset_name}_ratings.txt", 
            sep="\t", 
            names=["user_id", "item_id", "rating"]
        )
        social = pd.read_csv(
            raw_data_path / f"{dataset_name}_social.txt", 
            sep="\t", 
            names=["user_id", "friend_id"]
        )
        print("Raw data loaded successfully")
    except Exception as e:
        print(f"Error loading raw data: {str(e)}")
        return
    
    # 3. 构建用户和物品映射
    user2idx, item2idx, n_users, n_items = build_mappings(ratings)
    
    # 4. 构建矩阵和集合
    R, T, C, N, B = build_matrices_and_sets(
        ratings, social, user2idx, item2idx, n_users, n_items
    )
    
    # 5. 保存预处理数据
    save_json(user2idx, processed_data_path / "user2idx.json")
    save_json(item2idx, processed_data_path / "item2idx.json")
    save_sparse_matrix(R, processed_data_path / "R.npz")
    save_sparse_matrix(T, processed_data_path / "T.npz")
    save_pkl(C, processed_data_path / "C.pkl")
    save_pkl(N, processed_data_path / "N.pkl")
    save_pkl(B, processed_data_path / "B.pkl")
    
    # 6. 划分数据集
    observed_ratings = ratings[ratings["rating"] > 0].copy()
    observed_ratings["user_idx"] = observed_ratings["user_id"].map(user2idx)
    observed_ratings["item_idx"] = observed_ratings["item_id"].map(item2idx)
    
    # 只保留有有效映射的记录
    observed_ratings = observed_ratings.dropna(subset=["user_idx", "item_idx"])
    
    # 划分训练集、验证集和测试集
    train_ratings, temp_ratings = train_test_split(
        observed_ratings, test_size=0.2, random_state=42
    )
    val_ratings, test_ratings = train_test_split(
        temp_ratings, test_size=0.5, random_state=42
    )
    
    # 保存划分结果（只保留需要的列）
    train_ratings[["user_idx", "item_idx", "rating"]].to_csv(
        splits_path / f"{dataset_name}_train.csv", index=False
    )
    val_ratings[["user_idx", "item_idx", "rating"]].to_csv(
        splits_path / f"{dataset_name}_val.csv", index=False
    )
    test_ratings[["user_idx", "item_idx", "rating"]].to_csv(
        splits_path / f"{dataset_name}_test.csv", index=False
    )
    
    print(f"Preprocessing completed for {dataset_name} dataset!")
    print(f"Processed data saved to {processed_data_path}")
    print(f"Data splits saved to {splits_path}")

if __name__ == "__main__":
    # 预处理Ciao数据集
    preprocess_dataset("ciao")
    
    # 如需预处理Epinions数据集，取消下面一行的注释
    # preprocess_dataset("epinions")
