import torch

# 标题： 计算所有输入词元的注意力权重
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

# 3.4.1 逐步计算注意力权重
x_2 = inputs[1]
# 输入d_in的嵌入维度等于3， 输出d_out的嵌入维度等于2
d_in = inputs.shape[1]
d_out = 2

# 初始化3个权重矩阵，包括查询向量 W_query， 键向量 W_key，值向量 W_value
print("\n\n===初始化3个权重矩阵，包括查询向量W_query， 键向量W_key，值向量W_value")
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
print("W_query = ", W_query)
print("W_key = ", W_key)
print("W_value = ", W_value)

# W_query =  Parameter containing:
# tensor([[0.2961, 0.5166],
#         [0.2517, 0.6886],
#         [0.0740, 0.8665]])
# W_key =  Parameter containing:
# tensor([[0.1366, 0.1025],
#         [0.1841, 0.7264],
#         [0.3153, 0.6871]])
# W_value =  Parameter containing:
# tensor([[0.0756, 0.1966],
#         [0.3164, 0.4017],
#         [0.1186, 0.8274]])

# 计算查询向量， 键向量，值向量
print("\n\n===计算x_2的查询向量， 键向量，值向量")
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print("query_2 = ", query_2)
print("key_2 = ", key_2)
print("value_2 = ", value_2)
# query_2 =  tensor([0.4306, 1.4551])
# key_2 =  tensor([0.4433, 1.1419])
# value_2 =  tensor([0.3951, 1.0037])

# 通过矩阵乘法计算所有的键向量和值向量
print("\n\n===通过矩阵乘法计算所有输入向量的键向量和值向量")
keys = inputs @ W_key
values = inputs @ W_value
print("keys = ", keys)
print("values = ", values)
# 从输出可以看出，可以把三维度词元降低到二维
# keys =  tensor([[0.3669, 0.7646],
#         [0.4433, 1.1419],
#         [0.4361, 1.1156],
#         [0.2408, 0.6706],
#         [0.1827, 0.3292],
#         [0.3275, 0.9642]])
# values =  tensor([[0.1855, 0.8812],
#         [0.3951, 1.0037],
#         [0.3879, 0.9831],
#         [0.2393, 0.5493],
#         [0.1492, 0.3346],
#         [0.3221, 0.7863]])

# 计算注意力分数
print("\n\n===计算x_2的注意力分数")
keys_2 = keys[1]
attention_score_22 = query_2.dot(keys_2)
print("attention_score_22 = ", attention_score_22)
# attention_score_22 =  tensor(1.8524)

# 通过矩阵乘法计算所有的注意力分数(向量相似性， x_2点积x_i, i~6)
attention_score_2 = query_2 @ keys.T
print("attention_score_2 = ", attention_score_2)
# attention_score_2 =  tensor([1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440])

# 把注意力分数转换为注意力权重并归一化
print("\n===把注意力分数转换为注意力权重并归一化")
d_k = keys.shape[-1]
print("d_k = ", d_k)
# d_k =  2
attention_score_weights_2 = torch.softmax(attention_score_2 / d_k ** 0.5, dim=-1)
print("attention_score_weights_2 = ", attention_score_weights_2)
# attention_score_weights_2 =  tensor([0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820])

# 通过对值向量进行加权求和来计算上下文向量
# 其中， 注意力权重作为加权因子， 用于权衡每个值向量的重要性。
print("\n===通过对值向量进行加权求和来计算上下文向量")
context_vector_2 = attention_score_weights_2 @ values
print("context_vector_2 = ", context_vector_2)
# context_vector_2 =  tensor([0.3061, 0.8210])
