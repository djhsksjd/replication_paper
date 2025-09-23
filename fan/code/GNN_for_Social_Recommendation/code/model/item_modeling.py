import torch
import torch.nn as nn
import torch.nn.functional as F

class UserAggregation(nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim if hidden_dim else embedding_dim
        
        # 1. MLP g_u：融合p_t和e_r，生成f_jt（公式15）：输入维度2d，输出维度d
        self.g_u = nn.Sequential(
            nn.Linear(2 * embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, embedding_dim)
        )
        
        # 2. 注意力网络：计算μ_jt*（公式18）：输入维度2d，输出维度1
        self.attention_net = nn.Sequential(
            nn.Linear(2 * embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        
        # 聚合用的W和b（公式16中的参数）
        self.W_agg = nn.Linear(embedding_dim, embedding_dim)
        self.b_agg = nn.Parameter(torch.zeros(embedding_dim))
        nn.init.normal_(self.W_agg.weight, mean=0.0, std=0.1)
    
    def compute_f_jt(self, p_t, e_r):
        # p_t: (n_users, d) 或 (batch_size, d)，用户嵌入
        # e_r: (n_ratings, d) 或 (batch_size, d)，观点嵌入
        # 返回f_jt: (batch_size, d)，观点感知用户表示
        concat = torch.cat([p_t, e_r], dim=1)  # (batch_size, 2d)
        return self.g_u(concat)  # (batch_size, d)
    
    def forward(self, item_emb_qj, user_emb_p, opinion_emb_e, item_idx, B, R):
        # 输入参数：
        # item_emb_qj: (n_items, d)，所有物品的嵌入q_j
        # user_emb_p: (n_users, d)，所有用户的嵌入p_t
        # opinion_emb_e: (n_ratings, d)，所有观点的嵌入e_r
        # item_idx: (batch_size,)，当前批次的物品索引
        # B: list，B[j]是交互物品j的用户列表
        # R: 稀疏矩阵，用户-物品评分矩阵（用于获取用户t对物品j的评分r）
        
        batch_z_j = []
        for vj in item_idx:
            # 步骤1：获取交互物品j的用户列表B(j)和对应的评分
            users_t = B[vj]  # 用户t的索引列表，len = |B(j)|
            if not users_t:  # 若物品无交互用户，用0向量填充
                z_j = torch.zeros(self.embedding_dim, device=item_emb_qj.device)
                batch_z_j.append(z_j)
                continue
            
            # 获取用户t对物品j的评分r_tj（R[t, j]）
            ratings_tj = R[users_t, vj].data.toarray().squeeze()  # (|B(j)|,)
            rating_idx_tj = (ratings_tj - 1).astype(int)  # 转为0-based
            
            # 步骤2：生成所有f_jt（用户t的观点感知表示）
            p_t = user_emb_p[users_t]  # (|B(j)|, d)
            e_r = opinion_emb_e[rating_idx_tj]  # (|B(j)|, d)
            f_jt = self.compute_f_jt(p_t, e_r)  # (|B(j)|, d)
            
            # 步骤3：计算用户注意力权重μ_jt（公式18-19）
            qj = item_emb_qj[vj].unsqueeze(0).repeat(len(users_t), 1)  # (|B(j)|, d)
            concat_att = torch.cat([f_jt, qj], dim=1)  # (|B(j)|, 2d)
            mu_star = self.attention_net(concat_att).squeeze()  # (|B(j)|,)
            mu_jt = F.softmax(mu_star, dim=0)  # (|B(j)|,)，和为1
            
            # 步骤4：加权聚合得到z_j（公式17）
            weighted_f = mu_jt.unsqueeze(1) * f_jt  # (|B(j)|, d)
            sum_weighted_f = weighted_f.sum(dim=0)  # (d,)
            z_j = F.relu(self.W_agg(sum_weighted_f) + self.b_agg)  # (d,)
            batch_z_j.append(z_j)
        
        return torch.stack(batch_z_j, dim=0)  # (batch_size, d)
