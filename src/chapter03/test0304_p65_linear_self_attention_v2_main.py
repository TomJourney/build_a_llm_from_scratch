import torch
from src.chapter03.test0304_p65_linear_self_attention_v2 import SelfAttention_v2

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
print("\n===使用Pytorch线性层的自注意力python类(SelfAttention_v2)生成6个上下文向量")
torch.manual_seed(789)
self_attention_v2 = SelfAttention_v2(d_in, d_out)
print("self_attention_v2(inputs) = ", self_attention_v2(inputs))
# self_attention_v2(inputs) =  tensor([[-0.0739,  0.0713],
#         [-0.0748,  0.0703],
#         [-0.0749,  0.0702],
#         [-0.0760,  0.0685],
#         [-0.0763,  0.0679],
#         [-0.0754,  0.0693]], grad_fn=<MmBackward0>)

