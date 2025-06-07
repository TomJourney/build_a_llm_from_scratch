import torch
from src.chapter04.test0403_p96_feed_forward_module import FeedForward

print("\n\n===使用python字典指定小型GPT-2模型的配置")
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

feed_forward = FeedForward(GPT_CONFIG_124M)
# 创建批次维度为2的样本输入
x = torch.rand(2, 3, 768)
# 通过前馈神经网络处理输入样本
feed_forward_result = feed_forward(x)
print("\n\n===feed_forward_result.shape = ", feed_forward_result.shape)
# ===feed_forward_result.shape =  torch.Size([2, 3, 768])
print("\nfeed_forward_result = ", feed_forward_result)
# ffeed_forward_result =  tensor([[[-0.1190,  0.0430, -0.1174,  ..., -0.0706, -0.0469,  0.1185],
#          [-0.0381, -0.0049, -0.0023,  ..., -0.0143, -0.0321,  0.0842],
#          [ 0.0006,  0.0920, -0.0800,  ..., -0.0872, -0.0275,  0.1451]],
#
#         [[ 0.0026,  0.0888, -0.1051,  ...,  0.0077, -0.0346,  0.0587],
#          [-0.0164,  0.0680, -0.0986,  ..., -0.1227, -0.0268, -0.0614],
#          [ 0.0482,  0.0467, -0.1651,  ...,  0.0179,  0.0443,  0.0024]]],
#        grad_fn=<ViewBackward0>)

