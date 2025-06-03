import torch
import torch.nn as nn
# 标题： 实现一个简化的自注意力python类

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    # 前向传播
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attention_scores = queries @ keys.T # omega
        attention_scores_weights = torch.softmax(attention_scores/keys.shape[-1] ** 0.5, dim=-1) # 归一化
        context_vector = attention_scores_weights @ values  # 创建上下文向量
        return context_vector
