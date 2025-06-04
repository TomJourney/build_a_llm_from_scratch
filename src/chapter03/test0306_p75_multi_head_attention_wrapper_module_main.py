import torch

from src.chapter03.test0306_p75_multi_head_attention_wrapper_module import MultiHeadAttentionWrapper

## 标题： 实现一个简化的自注意力python类
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

# 步骤2： 使用多头注意力包装类创建上下文向量
print("\n\n=== 步骤2： 使用多头注意力包装类创建上下文向量")
torch.manual_seed(123)
context_length = batch_inputs.shape[1]  # 词元数量
d_in, d_out = 3, 2
# 实例化多头注意力对象
multi_head_attention = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
# 使用多头注意力对象创建上下文向量
context_vectors = multi_head_attention(batch_inputs)
print("\n===context_vectors.shape = ", context_vectors.shape)
# context_vectors.shape =  torch.Size([2, 6, 4])
print("\n===context_vectors = ", context_vectors)
# ===context_vectors =  tensor([[[-0.5337, -0.1051,  0.5085,  0.3508],
#          [-0.5323, -0.1080,  0.5084,  0.3508],
#          [-0.5323, -0.1079,  0.5084,  0.3506],
#          [-0.5297, -0.1076,  0.5074,  0.3471],
#          [-0.5311, -0.1066,  0.5076,  0.3446],
#          [-0.5299, -0.1081,  0.5077,  0.3493]],
#
#         [[-0.5337, -0.1051,  0.5085,  0.3508],
#          [-0.5323, -0.1080,  0.5084,  0.3508],
#          [-0.5323, -0.1079,  0.5084,  0.3506],
#          [-0.5297, -0.1076,  0.5074,  0.3471],
#          [-0.5311, -0.1066,  0.5076,  0.3446],
#          [-0.5299, -0.1081,  0.5077,  0.3493]]], grad_fn=<CatBackward0>)

