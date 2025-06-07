import torch
import torch.nn as nn

# 高效的多头注意力类
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert ((d_out % num_heads == 0), "d_out must be divisible by num_heads")

        self.d_out = d_out
        self.num_heads = num_heads
        # 减少投影维度以匹配所需的输出维度
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout) 
        self.register_buffer("mask",
                             torch.triu(torch.ones(context_length, context_length), diagonal=1)
                             )

    def forward(self, x):
        # 张量形状： (b, num_tokens, d_out)
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 通过添加一个 num_heads 维度来隐式分割矩阵
        # 然后展开最后一个维度： (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置： 从形状 (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算每个头的点积
        attention_score = queries @ keys.transpose(2, 3)
        # 被截断为词元数量的掩码
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # 使用掩码来填充注意力分数
        attention_score.masked_fill_(mask_bool, -torch.inf)
        # 对注意力分数做归一化并做丢失dropout处理
        attention_score_weights = torch.softmax(attention_score / keys.shape[-1] ** 0.5, dim=-1)
        attention_score_weights = self.dropout(attention_score_weights)

        # 张量形状: (b, num_tokens, n_heads, head_dim)
        context_vector = (attention_score_weights @ values).transpose(1, 2)

        # 组合头： 其中 self.d_out = self.num_heads * self.head_dim
        context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)
        # 添加一个可选的线性投影
        context_vector = self.out_proj(context_vector)
        return context_vector
