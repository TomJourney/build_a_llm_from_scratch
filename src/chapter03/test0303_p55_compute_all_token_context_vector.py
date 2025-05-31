import torch
from src.utils import MathUtis

# 标题： 计算所有输入词元的注意力权重
# 6个词元的嵌入向量表示
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

# 计算所有输入词元的注意力分数
print("\n\n=== 第1步：计算所有输入词元的注意力分数： 方法1-使用for循环")
all_attention_scores_1 = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        # 计算xi对xj的注意力分数（即xi对xj的相似度）
        all_attention_scores_1[i, j] = torch.dot(x_i, x_j)
print("all_attention_scores = ", all_attention_scores_1)

print("\n\n=== 第1步：计算所有输入词元的注意力分数： 方法2-使用矩阵乘法")
all_attention_scores_2 = inputs @ inputs.T
print("使用矩阵乘法计算词元间的注意力分数， all_attention_scores_2 = ", all_attention_scores_2)
# all_attention_scores_2 =  tensor([[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
#         [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
#         [0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
#         [0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
#         [0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
#         [0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450]])

print("\n\n 第2步：注意力分数归一化，得到所有词元的注意力权重")
all_attention_weights = torch.softmax(all_attention_scores_2, dim=-1)
print("all_attention_weights = ", all_attention_weights)
# all_attention_weights =  tensor([[0.2098, 0.2006, 0.1981, 0.1242, 0.1220, 0.1452],
#         [0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581],
#         [0.1390, 0.2369, 0.2326, 0.1242, 0.1108, 0.1565],
#         [0.1435, 0.2074, 0.2046, 0.1462, 0.1263, 0.1720],
#         [0.1526, 0.1958, 0.1975, 0.1367, 0.1879, 0.1295],
#         [0.1385, 0.2184, 0.2128, 0.1420, 0.0988, 0.1896]])

# all_attention_weights[1][2] 表示词元1与词元2间的相似性度量， all_attention_weights[3][2]表示词元3与词元2的相似性度量

print("\n\n 第3步：使用注意力权重通过矩阵乘法计算所有词元的上下文向量")
all_token_context_vectors = all_attention_weights @ inputs
print("all_token_context_vectors = ", all_token_context_vectors)

