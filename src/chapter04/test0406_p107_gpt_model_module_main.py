import tiktoken
import torch

from src.chapter04.test0406_p107_gpt_model_module import DiyGPTModel

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

print("\n=== 使用第2章介绍的tiktoken分词器对包含两个文本输入的批次进行分词处理， 以供GPT模型使用")
tokenizer = tiktoken.get_encoding("gpt2")
input_batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

print("\n=== tiktoken分词器分词结果")
print("tokenizer.encode(txt1) = ", tokenizer.encode(txt1))
print("tokenizer.encode(txt2) = ", tokenizer.encode(txt2))
# tokenizer.encode(txt1) =  [6109, 3626, 6100, 345]
# tokenizer.encode(txt2) =  [6109, 1110, 6622, 257]

input_batch.append(torch.tensor(tokenizer.encode(txt1)))
input_batch.append(torch.tensor(tokenizer.encode(txt2)))
input_batch = torch.stack(input_batch, dim=0)
print("\ninput_batch = ", input_batch)
# input_batch =  tensor([[6109, 3626, 6100,  345],
#         [6109, 1110, 6622,  257]])

print("\n\n===使用GPT模型处理输入词元")
torch.manual_seed(123)
diy_gpt_model = DiyGPTModel(GPT_CONFIG_124M)
diy_gpt_model_result = diy_gpt_model(input_batch)
print("\ndiy_gpt_model_result.shape = ", diy_gpt_model_result.shape)
print("\nGPT模型处理结果, diy_gpt_model_result = ", diy_gpt_model_result)
# diy_gpt_model_result.shape =  torch.Size([2, 4, 50257])
# diy_gpt_model_result =  tensor([[[ 0.3613,  0.4222, -0.0711,  ...,  0.3483,  0.4661, -0.2838],
#          [-0.1792, -0.5660, -0.9485,  ...,  0.0477,  0.5181, -0.3168],
#          [ 0.7120,  0.0332,  0.1085,  ...,  0.1018, -0.4327, -0.2553],
#          [-1.0076,  0.3418, -0.1190,  ...,  0.7195,  0.4023,  0.0532]],
#
#         [[-0.2564,  0.0900,  0.0335,  ...,  0.2659,  0.4454, -0.6806],
#          [ 0.1230,  0.3653, -0.2074,  ...,  0.7705,  0.2710,  0.2246],
#          [ 1.0558,  1.0318, -0.2800,  ...,  0.6936,  0.3205, -0.3178],
#          [-0.1565,  0.3926,  0.3288,  ...,  1.2630, -0.1858,  0.0388]]],
#        grad_fn=<UnsafeViewBackward0>)

print("\n\n===统计DiyGPT模型参数张量的总参数量")
total_params = sum(p.numel() for p in diy_gpt_model.parameters())
print(f"总参数量={total_params:,}")
# 总参数量=163,009,536

print("\n\n===计算GPTModel中1.63亿个参数的内存大小")
# 假设每个参数占用4个字节
total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"模型所需内存大小={total_size_mb:.2f} MB")
# 模型所需内存大小=621.83 MB

