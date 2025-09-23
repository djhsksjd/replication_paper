import torch
import torch.nn as nn

class RatingPrediction(nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=None, num_hidden_layers=3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim if hidden_dim else embedding_dim
        self.num_layers = num_hidden_layers
        
        # 构建MLP：输入维度2d（h_i ⊕ z_j），输出维度1（预测评分r'_ij）
        layers = []
        input_dim = 2 * embedding_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            input_dim = self.hidden_dim
        # 最终输出层：维度1（无激活，因评分是连续值）
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, h_i, z_j):
        # h_i: (batch_size, d)，用户最终因子
        # z_j: (batch_size, d)，物品最终因子
        concat = torch.cat([h_i, z_j], dim=1)  # (batch_size, 2d)
        r_hat = self.mlp(concat).squeeze()  # (batch_size,)，预测评分
        return r_hat
