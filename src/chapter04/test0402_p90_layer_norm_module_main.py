from src.chapter04.test0402_p90_layer_norm_module import LayerNorm
import torch
import torch.nn as nn

print("\n\n=== 层归一化")
torch.manual_seed(123)
# 步骤1： 创建2个训练样本，每个样本包含5个维度或特征
batch_example = torch.randn(2, 5)

layer_norm = LayerNorm(emb_dim=5)
layer_norm_result = layer_norm(batch_example)

layer_norm_result_mean = layer_norm_result.mean(dim=-1, keepdim=True)
layer_norm_result_variance = layer_norm_result.var(dim=-1, keepdim=True, unbiased=False)
print("层归一化结果均值mean = ", layer_norm_result_mean)
print("层归一化结果方差variance = ", layer_norm_result_variance)


