import torch
from src.chapter03.test0305_p72_simple_causal_attention_module import CausalAttention


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

# 步骤1：模拟批量输入
print("\n\n=== 步骤1：模拟批量输入")
batch_inputs = torch.stack((inputs, inputs), dim=0)
print("\nbatch.shape = ", batch_inputs.shape)
print("batch = ", batch_inputs)
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

# 第2步：使用因果注意力类CausalAttention生成6个上下文向量
print("\n===第2步：使用因果注意力类CausalAttention生成6个上下文向量")
torch.manual_seed(123)
context_length = batch_inputs.shape[1]
print("context_length = ", context_length)
# context_length =  6
causal_attention = CausalAttention(d_in, d_out, context_length, 0.0)
context_vectors = causal_attention(batch_inputs)
print("\n===context_vectors = ", context_vectors)
# ===context_vectors =  tensor([[[-0.5337, -0.1051],
#          [-0.5323, -0.1080],
#          [-0.5323, -0.1079],
#          [-0.5297, -0.1076],
#          [-0.5311, -0.1066],
#          [-0.5299, -0.1081]],
#
#         [[-0.5337, -0.1051],
#          [-0.5323, -0.1080],
#          [-0.5323, -0.1079],
#          [-0.5297, -0.1076],
#          [-0.5311, -0.1066],
#          [-0.5299, -0.1081]]], grad_fn=<UnsafeViewBackward0>)