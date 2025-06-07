import torch
import torch.nn as nn
from src.chapter03.test0306_p78_multi_head_attention_module import MultiHeadAttention

# 多头注意力类-MultiHeadAttention-测试用例

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

# 步骤1：模拟批量输入
print("\n\n=== 步骤1：模拟批量输入")
batch_inputs = torch.stack((inputs, inputs), dim=0)
print("\nbatch.shape = ", batch_inputs.shape)
print("\nbatch = ", batch_inputs)
# batch.shape =  torch.Size([2, 6, 3])
# batch =  tensor([[[0.4300, 0.1500, 0.8900],
#          [0.5500, 0.8700, 0.6600],
#          [0.5700, 0.8500, 0.6400],
#          [0.2200, 0.5800, 0.3300],
#          [0.7700, 0.2500, 0.1000],
#          [0.0500, 0.8000, 0.5500]],
#
#         [[0.4300, 0.1500, 0.8900],
#          [0.5500, 0.8700, 0.6600],
#          [0.5700, 0.8500, 0.6400],
#          [0.2200, 0.5800, 0.3300],
#          [0.7700, 0.2500, 0.1000],
#          [0.0500, 0.8000, 0.5500]]])

print("\n\n=== 步骤2： 使用多头注意力类计算上下文向量")
torch.manual_seed(123)
batch_size, context_length, d_in = batch_inputs.shape # 2, 6, 3
d_out = 2
multi_head_attention = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vector = multi_head_attention(batch_inputs)
print("\ncontext_vector = ", context_vector)
print("\nontext_vector.shape = ", context_vector.shape)
# context_vector =  tensor([[[0.3190, 0.4858],
#          [0.2943, 0.3897],
#          [0.2856, 0.3593],
#          [0.2693, 0.3873],
#          [0.2639, 0.3928],
#          [0.2575, 0.4028]],
#
#         [[0.3190, 0.4858],
#          [0.2943, 0.3897],
#          [0.2856, 0.3593],
#          [0.2693, 0.3873],
#          [0.2639, 0.3928],
#          [0.2575, 0.4028]]], grad_fn=<ViewBackward0>)
#
# ontext_vector.shape =  torch.Size([2, 6, 2])


