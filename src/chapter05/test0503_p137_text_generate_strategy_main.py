import tiktoken
import torch

from src.chapter04.test0406_p107_gpt_model_module import DiyGPTModel
from src.chapter04.test0406_p112_gpt_model_generate_text_module import generate_text_simple
from src.chapter05.test0501_p119_text_to_token_transfer_util_module import text_to_tokens_ids, token_ids_to_text

print("\n\n===使用python字典指定gpt模型的配置")
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

# 实例化gpt模型（使用自定义GPT模型）
torch.manual_seed(123)
# 创建自定义gpt模型实例
diy_gpt_model = DiyGPTModel(GPT_CONFIG_124M)

# 获取分词器
tokenizer = tiktoken.get_encoding("gpt2")

# 设置设备为cpu
diy_gpt_model.to("cpu")
# 设置为评估模型，以便关闭如dropout之类的随机组件
diy_gpt_model.eval()

# 使用大模型逐个生成文本
token_ids = generate_text_simple(gpt_model=diy_gpt_model,
                                 index_array=text_to_tokens_ids("Every effort moves you", tokenizer),
                                 max_new_tokens=25,
                                 context_size=GPT_CONFIG_124M["context_length"]
                                 )
print("生成的文本= ", token_ids_to_text(token_ids, tokenizer))
# 生成的文本=  Every effort moves you rentingetic wasnم refres RexMeCHicular stren Mortgage TT remember gard ACTIONSussedOND Land Engeleddedemate breaths proxies GalaxyForm

