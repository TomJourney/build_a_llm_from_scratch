import torch

from src.chapter03.test0304_p65_linear_self_attention_v2 import SelfAttention_v2

## 标题： 因果注意力的掩码实现
# 6个词元的嵌入向量表示
inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # x_1
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55]  # x_6
    ]
)

x_2 = inputs[1]
# 输入d_in的嵌入维度等于3， 输出d_out的嵌入维度等于2
d_in = inputs.shape[1]
d_out = 2

# 使用自注意力python类生成6个上下文向量
print("\n===使用Pytorch线性层的自注意力python类(SelfAttention_v2)")
torch.manual_seed(789)
self_attention_v2 = SelfAttention_v2(d_in, d_out)

# 因果注意力的掩码实现
print("\n\n========== 因果注意力的掩码实现")

# 步骤1：生成注意力权重
print("\n===步骤1：生成注意力权重")
queries = self_attention_v2.W_query(inputs)
keys = self_attention_v2.W_key(inputs)
attention_scores = queries @ keys.T
attention_scores_weights = torch.softmax(attention_scores / keys.shape[-1] * 0.5, dim=-1)
print("attention_scores_weights = ", attention_scores_weights)
# attention_scores_weights =  tensor([[0.1754, 0.1661, 0.1663, 0.1626, 0.1687, 0.1610],
#         [0.1793, 0.1666, 0.1667, 0.1606, 0.1668, 0.1599],
#         [0.1791, 0.1666, 0.1667, 0.1607, 0.1668, 0.1600],
#         [0.1736, 0.1667, 0.1668, 0.1633, 0.1665, 0.1630],
#         [0.1723, 0.1668, 0.1668, 0.1639, 0.1664, 0.1638],
#         [0.1758, 0.1667, 0.1667, 0.1623, 0.1667, 0.1618]],
#        grad_fn=<SoftmaxBackward0>)

# 步骤2： 创建掩码矩阵
print("\n===步骤2： 创建掩码矩阵（对角线以上为0的掩码）")
context_length = attention_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print("\n===掩码矩阵：mask_simple = ", mask_simple)
# 掩码矩阵：mask_simple =  tensor([[1., 0., 0., 0., 0., 0.],
#         [1., 1., 0., 0., 0., 0.],
#         [1., 1., 1., 0., 0., 0.],
#         [1., 1., 1., 1., 0., 0.],
#         [1., 1., 1., 1., 1., 0.],
#         [1., 1., 1., 1., 1., 1.]])

# 步骤3：使用掩码矩阵与注意力权重矩阵相乘，使对角线上方的值变为0
print("\n\n===步骤3：使用掩码矩阵与注意力权重矩阵相乘，使对角线上方的值变为0")
mask_simple = attention_scores_weights * mask_simple
print("\n===mask_simple = ", mask_simple)
# ===mask_simple =  tensor([[0.1754, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.1793, 0.1666, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.1791, 0.1666, 0.1667, 0.0000, 0.0000, 0.0000],
#         [0.1736, 0.1667, 0.1668, 0.1633, 0.0000, 0.0000],
#         [0.1723, 0.1668, 0.1668, 0.1639, 0.1664, 0.0000],
#         [0.1758, 0.1667, 0.1667, 0.1623, 0.1667, 0.1618]],
#        grad_fn=<MulBackward0>)


# 步骤4： 重新归一化注意力权重， 使每一行的总和再次为1.
print("\n\n===步骤4： 重新归一化注意力权重， 使每一行的总和再次为1.")
row_nums = mask_simple.sum(dim=-1, keepdim=True)
mask_simple_norm = mask_simple / row_nums
print("\n=== 掩码后的归一化注意力权重矩阵 mask_simple_norm = ", mask_simple_norm)
# 归一化后的注意力权重矩阵 mask_simple_norm =  tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.5183, 0.4817, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.3495, 0.3251, 0.3254, 0.0000, 0.0000, 0.0000],
#         [0.2590, 0.2487, 0.2488, 0.2435, 0.0000, 0.0000],
#         [0.2061, 0.1994, 0.1995, 0.1960, 0.1990, 0.0000],
#         [0.1758, 0.1667, 0.1667, 0.1623, 0.1667, 0.1618]],
#        grad_fn=<DivBackward0>)

## 步骤5： 对注意力权重矩阵掩码进行优化：
## 通过创建一个对角线以上是1的掩码，并将这些1替换为负无穷大（-inf），来实现更高效的掩码方法
print("\n\n=== 步骤5： 通过创建一个对角线以上是1的掩码，并将这些1替换为负无穷大（-inf），来实现更高效的掩码方法")
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attention_scores.masked_fill(mask.bool(), -torch.inf)
print("\nmasked = ", masked)
# masked =  tensor([[0.2899,   -inf,   -inf,   -inf,   -inf,   -inf],
#         [0.4656, 0.1723,   -inf,   -inf,   -inf,   -inf],
#         [0.4594, 0.1703, 0.1731,   -inf,   -inf,   -inf],
#         [0.2642, 0.1024, 0.1036, 0.0186,   -inf,   -inf],
#         [0.2183, 0.0874, 0.0882, 0.0177, 0.0786,   -inf],
#         [0.3408, 0.1270, 0.1290, 0.0198, 0.1290, 0.0078]],
#        grad_fn=<MaskedFillBackward0>)

## 步骤6： 对上述掩码矩阵应用softmax函数归一化即可
print("\n\n=== 步骤6： 对上述掩码矩阵应用softmax函数归一化即可")
attention_scores_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=-1)
print("=== attention_scores_weights = ", attention_scores_weights)
# === attention_scores_weights =  tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
#         [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
#         [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
#         [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
#        grad_fn=<SoftmaxBackward0>)

