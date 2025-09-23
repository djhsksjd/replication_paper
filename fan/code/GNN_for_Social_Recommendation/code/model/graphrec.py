import torch
import torch.nn as nn
from .embeddings import GraphRecEmbeddings
from .user_modeling import ItemAggregation, SocialAggregation, UserFactorFusion
from .item_modeling import UserAggregation
from .prediction import RatingPrediction

class GraphRec(nn.Module):
    def __init__(self, n_users, n_items, n_ratings=5, embedding_dim=64, hidden_dim=64, num_hidden_layers=3):
        super().__init__()
        # 1. 嵌入层
        self.embeddings = GraphRecEmbeddings(n_users, n_items, n_ratings, embedding_dim)
        
        # 2. 用户建模模块
        self.item_agg = ItemAggregation(embedding_dim, hidden_dim)
        self.item_agg.init_agg_params()  # 初始化物品聚合的W和b
        self.social_agg = SocialAggregation(embedding_dim, hidden_dim)
        self.user_fusion = UserFactorFusion(embedding_dim, hidden_dim, num_hidden_layers)
        
        # 3. 物品建模模块
        self.user_agg = UserAggregation(embedding_dim, hidden_dim)
        
        # 4. 评分预测模块
        self.rating_pred = RatingPrediction(embedding_dim, hidden_dim, num_hidden_layers)
    
    def forward(self, user_idx, item_idx, C, N, B, R):
        # 输入：当前批次的用户索引、物品索引，及核心集合/矩阵
        # 步骤1：生成所有嵌入向量
        p_i_all = self.embeddings.user_emb.weight  # (n_users, d)
        q_j_all = self.embeddings.item_emb.weight  # (n_items, d)
        e_r_all = self.embeddings.opinion_emb.weight  # (n_ratings, d)
        
        # 步骤2：用户建模：计算h_i（批量处理用户）
        # 2.1 物品聚合：计算当前批次用户的h_I
        h_I_batch = self.item_agg(p_i_all, q_j_all, e_r_all, user_idx, C, R)  # (batch_size, d)
        # 2.2 预计算所有用户的h_I（用于社交聚合，因需好友的h_O^I）
        h_I_all = self.item_agg(p_i_all, q_j_all, e_r_all, torch.arange(p_i_all.shape[0], device=p_i_all.device), C, R)  # (n_users, d)
        # 2.3 社交聚合：计算当前批次用户的h_S
        h_S_batch = self.social_agg(p_i_all, h_I_all, user_idx, N)  # (batch_size, d)
        # 2.4 融合h_I和h_S得到h_i
        h_i_batch = self.user_fusion(h_I_batch, h_S_batch)  # (batch_size, d)
        
        # 步骤3：物品建模：计算当前批次物品的z_j
        z_j_batch = self.user_agg(q_j_all, p_i_all, e_r_all, item_idx, B, R)  # (batch_size, d)
        
        # 步骤4：评分预测
        r_hat_batch = self.rating_pred(h_i_batch, z_j_batch)  # (batch_size,)
        
        return r_hat_batch
