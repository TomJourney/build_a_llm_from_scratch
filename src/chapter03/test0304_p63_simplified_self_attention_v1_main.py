import torch
from src.chapter03.test0304_p63_simplified_self_attention_v1 import SelfAttention_v1

## 标题： 实现一个简化的自注意力python类
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

x_2 = inputs[1]
# 输入d_in的嵌入维度等于3， 输出d_out的嵌入维度等于2
d_in = inputs.shape[1]
d_out = 2

# 使用自注意力python类生成6个上下文向量
print("\n===使用自注意力python类(SelfAttention_v1)生成6个上下文向量")
torch.manual_seed(123)
self_attention_v1 = SelfAttention_v1(d_in, d_out)
print(self_attention_v1(inputs))

# tensor([[0.2996, 0.8053],
#         [0.3061, 0.8210],
#         [0.3058, 0.8203],
#         [0.2948, 0.7939],
#         [0.2927, 0.7891],
#         [0.2990, 0.8040]], grad_fn=<MmBackward0>)
#

