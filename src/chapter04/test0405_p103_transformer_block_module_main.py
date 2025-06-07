import torch
from src.chapter04.test0405_p103_transformer_block_module import TransformerBlock

# Transformer块测试案例

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

print("\n\n=== Transformer块测试案例")
torch.manual_seed(123)
x = torch.rand(2, 4, 768)
print("\nx.shape = ", x.shape)
# x.shape =  torch.Size([2, 4, 768])

transformer_block = TransformerBlock(GPT_CONFIG_124M)
transformer_block_output = transformer_block(x)
print("\ntransformer_block_output.shape = ", transformer_block_output.shape)
print("\ntransformer_block_output = ", transformer_block_output)
# transformer_block_output.shape =  torch.Size([2, 4, 768])
# transformer_block_output =  tensor([[[ 0.3628,  0.2068,  0.1378,  ...,  1.6130,  0.6834,  0.9405],
#          [ 0.2828, -0.1074,  0.0276,  ...,  1.3251,  0.3856,  0.7150],
#          [ 0.5910,  0.4426,  0.3541,  ...,  1.5575,  0.7260,  1.2165],
#          [ 0.2230,  0.7529,  0.9257,  ...,  0.9274,  0.7475,  0.9625]],
#
#         [[ 0.3897,  0.8890,  0.6291,  ...,  0.4641,  0.3794,  0.1366],
#          [ 0.0259,  0.4077, -0.0179,  ...,  0.7759,  0.5887,  0.7169],
#          [ 0.8902,  0.2369,  0.1605,  ...,  0.9420,  0.8058,  0.5586],
#          [ 0.4029,  0.4937,  0.4106,  ...,  1.7933,  1.3422,  0.6940]]],
#        grad_fn=<AddBackward0>)

