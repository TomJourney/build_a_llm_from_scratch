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


# 使用pytorch.softmax函数进行朴素归一化
print("\n=====================使用pytorch.softmax函数对x2的注意力分数进行朴素归一化， 得到x2的注意力权重")
attention_weight_2_normalization_torch = torch.softmax(attention_score_2, dim=0)
print("attention_weight_2_normalization_torch = ", attention_weight_2_normalization_torch)
print("attention_weight_2_normalization_torch.sum() = ", attention_weight_2_normalization_torch.sum())
# attention_weight_2_normalization_torch =  tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
# attention_weight_2_normalization_torch.sum() =  tensor(1.)

print("\n\n==================== 计算x2的上下文向量z2")
# 把第2个元素作为查询词元
query = inputs[1]
context_vector_2 = torch.zeros(query.shape)
print("context_vector_2 = ", context_vector_2)
# context_vector_2 =  tensor([0., 0., 0.])

for i, x_i in enumerate(inputs):
    print("\n====================  我是行分隔符 ====================")
    # x_i对x2的权值（注意力权重） 乘上 x_i的嵌入向量，并累加后得到对应x2的上下文向量 context_vector_2
    context_vector_2 += attention_weight_2_normalization_torch[i] * x_i
print("context_vector_2 = ", context_vector_2)
# context_vector_2 =  tensor([0.4419, 0.6515, 0.5683])


