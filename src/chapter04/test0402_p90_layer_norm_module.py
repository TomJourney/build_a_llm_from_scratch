import torch
import torch.nn as nn

print("\n\n=== 层归一化")
torch.manual_seed(123)
# 步骤1： 创建2个训练样本，每个样本包含5个维度或特征
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
layer_out = layer(batch_example)
print("层处理（输出）结果-layer_out = ", layer_out)

# 步骤2： 上述编写的神经网络层包括一个线性层和一个非线性激活函数ReLU（修正线性单元）， ReLU是神经网络中的一种标准激活函数
print("\n=== 层归一化前，检查层处理结果的均值与方差")
layer_out_mean = layer_out.mean(dim=-1, keepdim=True)
layer_out_variance = layer_out.var(dim=-1, keepdim=True)
print("均值mean = ", layer_out_mean, "\n方差variance = ", layer_out_variance)

# 步骤3：层归一化操作：具体方法是减少均值，并将结果除以方差的平方根（即标准差）
print("\n===步骤3：层归一化操作：具体方法是减少均值，并将结果除以方差的平方根（即标准差）")
layer_out_norm = (layer_out - layer_out_mean) / torch.sqrt(layer_out_variance)
layer_out_norm_mean = layer_out_norm.mean(dim=-1, keepdim=True)
layer_out_norm_variance = layer_out_norm.var(dim=-1, keepdim=True)
print("层归一化结果 layer_out_norm = ", layer_out_norm)
print("层归一化结果均值 layer_out_norm_mean = ", layer_out_norm_mean)
print("层归一化结果方差 layer_out_norm_variance = ", layer_out_norm_variance)

# 为提高可读性， 通过将sci_mode设置为False来关闭科学计数法，从而在打印张量值时避免使用科学计数法
print("\n\n=== 通过将sci_mode设置为False来关闭科学计数法") 
torch.set_printoptions(sci_mode=False)
print("层归一化结果均值 layer_out_norm_mean = ", layer_out_norm_mean)
print("层归一化结果方差 layer_out_norm_variance = ", layer_out_norm_variance)



