import torch
import torch.nn as nn
import torch.nn.functional as F

class ItemAggregation(nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim if hidden_dim else embedding_dim
        
        # 1. MLP g_v：融合q_a和e_r，生成x_ia（公式2）：输入维度2d，输出维度d
        self.g_v = nn.Sequential(
            nn.Linear(2 * embedding_dim, self.hidden_dim),
            nn.ReLU(),  # 论文用ReLU激活（σ为rectified linear unit）
            nn.Linear(self.hidden_dim, embedding_dim)
        )
        
        # 2. 注意力网络：计算α_ia*（公式5）：输入维度2d，输出维度1
        self.attention_net = nn.Sequential(
            nn.Linear(2 * embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
    
    def compute_x_ia(self, q_a, e_r):
        # q_a: (n_items, d) 或 (batch_size, d)，物品嵌入
        # e_r: (n_ratings, d) 或 (batch_size, d)，观点嵌入
        # 返回x_ia: (batch_size, d)，观点感知交互表示
        concat = torch.cat([q_a, e_r], dim=1)  # (batch_size, 2d)
        return self.g_v(concat)  # (batch_size, d)
    
    def forward(self, user_emb_pi, item_emb_q, opinion_emb_e, user_idx, C, R):
        # 输入参数：
        # user_emb_pi: (n_users, d)，所有用户的嵌入p_i
        # item_emb_q: (n_items, d)，所有物品的嵌入q_j
        # opinion_emb_e: (n_ratings, d)，所有观点的嵌入e_r
        # user_idx: (batch_size,)，当前批次的用户索引
        # C: list，C[i]是用户i交互的物品列表
        # R: 稀疏矩阵，用户-物品评分矩阵（用于获取用户i对物品a的评分r）
        
        batch_h_I = []
        for u in user_idx:
            # 步骤1：获取用户u交互的物品列表C(u)和对应的评分
            items_a = C[u]  # 物品a的索引列表，len = |C(u)|
            if not items_a:  # 若用户无交互物品，用0向量填充（实际数据中极少）
                h_I_u = torch.zeros(self.embedding_dim, device=user_emb_pi.device)
                batch_h_I.append(h_I_u)
                continue
            
            # 获取用户u对物品a的评分r_ua（R[u, a]）
            ratings_ua = R[u, items_a].data.toarray().squeeze()  # (|C(u)|,)
            rating_idx_ua = (ratings_ua - 1).astype(int)  # 转为0-based（1→0，5→4）
            
            # 步骤2：生成所有x_ia（物品a的观点感知表示）
            q_a = item_emb_q[items_a]  # (|C(u)|, d)
            e_r = opinion_emb_e[rating_idx_ua]  # (|C(u)|, d)
            x_ia = self.compute_x_ia(q_a, e_r)  # (|C(u)|, d)
            
            # 步骤3：计算注意力权重α_ia（公式5-6）
            pi = user_emb_pi[u].unsqueeze(0).repeat(len(items_a), 1)  # (|C(u)|, d)
            concat_att = torch.cat([x_ia, pi], dim=1)  # (|C(u)|, 2d)
            alpha_star = self.attention_net(concat_att).squeeze()  # (|C(u)|,)
            alpha_ia = F.softmax(alpha_star, dim=0)  # (|C(u)|,)，和为1
            
            # 步骤4：加权聚合得到h_I_u（公式4）
            weighted_x = alpha_ia.unsqueeze(1) * x_ia  # (|C(u)|, d)
            sum_weighted_x = weighted_x.sum(dim=0)  # (d,)
            # 论文公式1中的W和b（线性变换+ReLU）：此处用一层线性层实现
            h_I_u = F.relu(self.W_agg(sum_weighted_x) + self.b_agg)  # (d,)
            batch_h_I.append(h_I_u)
        
        # 拼接批次内所有用户的h_I
        return torch.stack(batch_h_I, dim=0)  # (batch_size, d)
    
    # 补充：初始化聚合用的W和b（公式1中的参数）
    def init_agg_params(self):
        self.W_agg = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.b_agg = nn.Parameter(torch.zeros(self.embedding_dim))
        nn.init.normal_(self.W_agg.weight, mean=0.0, std=0.1)


class SocialAggregation(nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim if hidden_dim else embedding_dim
        
        # 注意力网络：计算β_io*（公式10）：输入维度2d，输出维度1
        self.attention_net = nn.Sequential(
            nn.Linear(2 * embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        
        # 聚合用的W和b（公式7中的参数）
        self.W_agg = nn.Linear(embedding_dim, embedding_dim)
        self.b_agg = nn.Parameter(torch.zeros(embedding_dim))
        nn.init.normal_(self.W_agg.weight, mean=0.0, std=0.1)
    
    def forward(self, user_emb_pi, h_I_all, user_idx, N):
        # 输入参数：
        # user_emb_pi: (n_users, d)，所有用户的嵌入p_i
        # h_I_all: (n_users, d)，所有用户的物品空间因子h_I
        # user_idx: (batch_size,)，当前批次的用户索引
        # N: list，N[i]是用户i的社交好友列表
        
        batch_h_S = []
        for u in user_idx:
            # 步骤1：获取用户u的社交好友列表N(u)
            friends_o = N[u]  # 用户o的索引列表，len = |N(u)|
            if not friends_o:  # 若用户无社交好友，用0向量填充
                h_S_u = torch.zeros(self.embedding_dim, device=user_emb_pi.device)
                batch_h_S.append(h_S_u)
                continue
            
            # 步骤2：获取好友o的物品空间因子h_O^I
            h_O_I = h_I_all[friends_o]  # (|N(u)|, d)
            
            # 步骤3：计算社交注意力权重β_io（公式10-11）
            pi = user_emb_pi[u].unsqueeze(0).repeat(len(friends_o), 1)  # (|N(u)|, d)
            concat_att = torch.cat([h_O_I, pi], dim=1)  # (|N(u)|, 2d)
            beta_star = self.attention_net(concat_att).squeeze()  # (|N(u)|,)
            beta_io = F.softmax(beta_star, dim=0)  # (|N(u)|,)，和为1
            
            # 步骤4：加权聚合得到h_S_u（公式9）
            weighted_h_O = beta_io.unsqueeze(1) * h_O_I  # (|N(u)|, d)
            sum_weighted_h = weighted_h_O.sum(dim=0)  # (d,)
            h_S_u = F.relu(self.W_agg(sum_weighted_h) + self.b_agg)  # (d,)
            batch_h_S.append(h_S_u)
        
        return torch.stack(batch_h_S, dim=0)  # (batch_size, d)


class UserFactorFusion(nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=None, num_hidden_layers=3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim if hidden_dim else embedding_dim
        self.num_layers = num_hidden_layers
        
        # 构建MLP：输入维度2d（h_I ⊕ h_S），输出维度d（h_i）
        layers = []
        input_dim = 2 * embedding_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))  # 论文2.6节：用Dropout缓解过拟合
            input_dim = self.hidden_dim
        # 最终输出层：维度d
        layers.append(nn.Linear(input_dim, embedding_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, h_I, h_S):
        # h_I: (batch_size, d)，物品空间因子
        # h_S: (batch_size, d)，社交空间因子
        concat = torch.cat([h_I, h_S], dim=1)  # (batch_size, 2d)
        h_i = self.mlp(concat)  # (batch_size, d)
        return h_i
