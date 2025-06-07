import torch
import torch.nn as nn

# 批处理矩阵乘法
a = torch.tensor([
                    [
                        [
                            [0.2745, 0.6584, 0.2775, 0.8573],
                            [0.8993, 0.0390, 0.9268, 0.7388],
                            [0.7179, 0.7058, 0.9156, 0.4340]
                        ],
                        [
                            [0.0772, 0.3565, 0.1479, 0.5331],
                            [0.4066, 0.2318, 0.4545, 0.9737],
                            [0.4606, 0.5159, 0.4220, 0.5786]
                        ]
                    ]
                 ]
)

# 在原始张量与转置后的张量之间执行批处理矩阵乘法， 其中我们转置了最后两个维度， 即 num_tokens 和 head_dim
print("\n\n=== 在原始张量与转置后的张量之间执行批处理矩阵乘法")
print("\na.transpose(2, 3) = ", a.transpose(2, 3))
result = a @ a.transpose(2, 3)
print("\n乘法结果 result = ", result)

# 单独计算每个头的矩阵乘法
print("\n\n=== 单独计算每个头的矩阵乘法（最终结果与批处理矩阵乘法结果完全相同）")
first_head = a[0, 0, :, :]
first_result = first_head @ first_head.T
print("\nfirst_result = ", first_result)

second_head = a[0, 1, :, :]
second_result = second_head @ second_head.T
print("\nsecond_result = ", second_result)



