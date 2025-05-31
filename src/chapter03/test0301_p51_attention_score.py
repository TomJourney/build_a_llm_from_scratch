import torch
from src.utils import MathUtis

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89], # x_1
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55] # x_6
    ]
)

print("\n\n=== 计算注意力分数")
# 把第2个元素作为查询词元
query = inputs[1]
print("inputs.shape = ", inputs.shape)
# inputs.shape =  torch.Size([6, 3])

print("inputs.shape[0] = ", inputs.shape[0])
# inputs.shape[0] =  6

attention_score_2 = torch.empty(inputs.shape[0])
print("attention_score_2 = ", attention_score_2)
# attention_score_2 =  tensor([-8.3727e+34,  7.5530e-43,  0.0000e+00,  0.0000e+00,  0.0000e+00, 0.0000e+00])

for i, x_i in enumerate(inputs):
    attention_score_2[i] = torch.dot(x_i, query)
print("after torch.dot(), x2的注意力分数=attention_score_2 = ", attention_score_2)
# after torch.dot(), attention_score_2 =  tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])

# 对注意力分数归一化得到注意力权重
print("\n\n===对注意力分数归一化得到注意力权重")
attention_weight_2_temp = attention_score_2 / attention_score_2.sum()
print("attention_weight_2_temp = ", attention_weight_2_temp)
print("attention_weight_2_temp.sum() = ", attention_weight_2_temp.sum())
# attention_weight_2_temp =  tensor([0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656])
# attention_weight_2_temp.sum() =  tensor(1.0000)

# 使用自定义softmax函数进行朴素归一化
print("\n\n===使用自定义softmax函数进行朴素归一化")
attention_weight_2_naive = MathUtis.diy_softmax_naive(attention_score_2)
print("attention_weight_2_naive = ", attention_weight_2_naive)
print("attention_weight_2_naive.sum() = ", attention_weight_2_naive.sum())
# attention_weight_2_naive =  tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
# attention_weight_2_naive.sum() =  tensor(1.)

# 使用pytorch.softmax函数进行朴素归一化
print("\n===使用pytorch.softmax函数进行朴素归一化")
attention_weight_2_torch = torch.softmax(attention_score_2, dim=0)
print("attention_weight_2_torch = ", attention_weight_2_torch)
print("attention_weight_2_torch.sum() = ", attention_weight_2_torch.sum())
# attention_weight_2_torch =  tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
# attention_weight_2_torch.sum() =  tensor(1.)

