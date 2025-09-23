import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphRecEmbeddings(nn.Module):
    def __init__(self, n_users, n_items, n_ratings=5, embedding_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim  # d，嵌入向量长度
        # 1. 用户嵌入p_i: (n_users, d)
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        # 2. 物品嵌入q_j: (n_items, d)
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        # 3. 观点嵌入e_r: (n_ratings, d)（n_ratings=5，对应1-5星）
        self.opinion_emb = nn.Embedding(n_ratings, embedding_dim)
        
        # 初始化嵌入（论文2.6节：高斯分布N(0, 0.1)）
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.opinion_emb.weight, mean=0.0, std=0.1)
    
    def forward(self, user_idx, item_idx, rating_idx):
        # user_idx: (batch_size,)，用户索引
        # item_idx: (batch_size,)，物品索引
        # rating_idx: (batch_size,)，评分索引（1-5星对应0-4，因Embedding需0-based）
        p_i = self.user_emb(user_idx)  # (batch_size, d)
        q_j = self.item_emb(item_idx)  # (batch_size, d)
        e_r = self.opinion_emb(rating_idx)  # (batch_size, d)
        return p_i, q_j, e_r