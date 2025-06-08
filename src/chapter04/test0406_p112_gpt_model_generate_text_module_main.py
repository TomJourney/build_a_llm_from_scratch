import tiktoken
import torch

from src.chapter04.test0406_p107_gpt_model_module import DiyGPTModel
from src.chapter04.test0406_p112_gpt_model_generate_text_module import generate_text_simple

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

# 测试案例-使用generate_text_simple生成文本（生成下一个词元）
start_context = "Hello, I am"

print("\n=== 使用第2章介绍的tiktoken分词器对包含两个文本输入的批次进行分词处理， 以供GPT模型使用")
tokenizer = tiktoken.get_encoding("gpt2")
encoded_token_ids = tokenizer.encode(start_context)
print("编码后的词元id = ", encoded_token_ids)
# 编码后的词元id =  [15496, 11, 314, 716]

# 添加batch维度
encoded_tensor = torch.tensor(encoded_token_ids).unsqueeze(0)
print("编码后的张量形状，encoded_tensor.shape = ", encoded_tensor.shape)
# encoded_tensor.shape =  torch.Size([1, 4])

print("\n\n===使用GPT模型处理输入词元")
torch.manual_seed(123)
diy_gpt_model = DiyGPTModel(GPT_CONFIG_124M)
# 模型设置为 .eval()模式
diy_gpt_model.eval()

# 对编码后的输入张量使用 generate_text_simple 函数
next_token_predict_result = generate_text_simple(
    gpt_model=diy_gpt_model,
    index_array=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("next_token_predict_result = ", next_token_predict_result)
print("len(next_token_predict_result[0]) = ", len(next_token_predict_result[0]))
# next_token_predict_result =  tensor([[15496,    11,   314,   716, 27018, 24086, 47843, 30961, 42348,  7267]])
# len(next_token_predict_result[0]) =  10

# 使用分词器的解码方法.decode()把id转为文本
decoded_text = tokenizer.decode(next_token_predict_result.squeeze(0).tolist())
print("解密后的文本， decoded_text = ", decoded_text)
# decoded_text =  Hello, I am Featureiman Byeswickattribute argue
