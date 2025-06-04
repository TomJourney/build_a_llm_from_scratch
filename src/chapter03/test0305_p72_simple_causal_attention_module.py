import torch
import torch.nn as nn

## 标题： 实现一个简化的因果注意力类 (Causal Attention)
# 使用因果注意力与dropout修改自注意力类，修改后的类将成为开发多头注意力的基础
# 而多头注意力是我们最终实现的注意力类

class CausalAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 与之前的注意力类相比，多了一个dropout层
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),
            diagonal=1)
        )
    # 前向传播
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 将维度1与维度2转置， 将批维度保持在第一个位置（0）
        attention_scores = queries @ keys.transpose(1, 2)

        # 在pytorch中， 所有以下划线结尾的操作都会直接作用于原数据， 从而减少不必要的内存复制
        attention_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attention_scores_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )

        attention_scores_weights = self.dropout(attention_scores_weights)
        context_vector = attention_scores_weights @ values
        return context_vector