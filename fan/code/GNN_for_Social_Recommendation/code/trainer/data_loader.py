import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class RatingDataset(Dataset):
    def __init__(self, ratings_df):
        self.user_idx = ratings_df["user_idx"].values
        self.item_idx = ratings_df["item_idx"].values
        self.rating = ratings_df["rating"].values
    
    def __len__(self):
        return len(self.user_idx)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.user_idx[idx], dtype=torch.long),
            torch.tensor(self.item_idx[idx], dtype=torch.long),
            torch.tensor(self.rating[idx], dtype=torch.float32)
        )

def get_dataloader(train_path, val_path, test_path, batch_size=128):
    """
    获取训练、验证和测试数据的DataLoader
    """
    # 读取CSV文件
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # 创建数据集
    train_dataset = RatingDataset(train_df)
    val_dataset = RatingDataset(val_df)
    test_dataset = RatingDataset(test_df)
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
