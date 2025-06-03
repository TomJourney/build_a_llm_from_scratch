import torch
import torch.nn as nn
# 标题： 使用Pytorch线性层的自注意力类

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    # 前向传播
    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attention_scores = queries @ keys.T # omega
        attention_scores_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)  # 归一化
        context_vector = attention_scores_weights @ values  # 创建上下文向量
        return context_vector
