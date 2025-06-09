import torch
import tiktoken
from src.chapter04.test0406_p107_gpt_model_module import DiyGPTModel
from src.chapter04.test0406_p112_gpt_model_generate_text_module import generate_text_simple
from src.chapter05.test0501_p119_text_to_token_transfer_util_module import text_to_tokens_ids, token_ids_to_text

# 文本到词元id转换的工具函数-测试案例
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
gpt_model = DiyGPTModel(GPT_CONFIG_124M)
gpt_model.eval()

# 使用文本到词元id转换的工具函数，生成文本
start_context = "Every effort moves you"
# 获取分词器
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    gpt_model=gpt_model,
    index_array=text_to_tokens_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("词元转换为文本工具函数结果 = ", token_ids_to_text(token_ids, tokenizer))
# 词元转换为文本工具函数结果 =  Every effort moves you rentingetic wasnم refres RexMeCHicular stren