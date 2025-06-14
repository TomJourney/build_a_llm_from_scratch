import tiktoken
import torch.nn

from gpt_download import download_and_load_gpt2
from src.chapter04.test0406_p107_gpt_model_module import DiyGPTModel
from src.chapter05.test0501_p119_text_to_token_transfer_util_module import text_to_tokens_ids, token_ids_to_text
from src.chapter05.test0503_p142_modify_text_generate_function import based_temperature_topk_generate_text_simple
from src.chapter05.test0505_p148_load_gpt2_params_to_diy_gpt_model_module import load_weights_into_gpt

# 应用GPT-2模型参数到DiyGPTModel

# 通过download_and_load_gpt2函数导入
print("\n step2：使用 download_and_load_gpt2函数 加载gpt-2架构设置和权重参数到python会话中")
gpt2_settings, gpt2_params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2", is_download=False
)

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

# 不同大小的GPT2大模型的参数量从1.24亿到15.58亿不等。
# 核心架构相同，唯一区别是嵌入层大小-emb_dim， 以及诸如注意力头-n_heads， Transformer块的重复次数-n_layers不同
# 字典保存不同模型尺寸的GPT模型参数
gpt2_model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (744M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25}
}
# 选择gpt2-small模型参数
model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(gpt2_model_configs[model_name])
# OpenAI的原始GPT-2模型使用1024个词元长度进行训练
NEW_CONFIG.update({"context_length": 1024})
# OpenAI在多头注意力模块的线性层中使用偏置向量实现查询矩阵，键矩阵和值矩阵的计算
NEW_CONFIG.update({"qkv_bias": True})

# 创建diy大模型实例
diy_gpt_model_based_gpt2_config = DiyGPTModel(NEW_CONFIG)
diy_gpt_model_based_gpt2_config.eval()

# GPT模型实例使用随机权重初始化，所以最后一步，我们需要用gpt2模型的权重覆盖随机权重
load_weights_into_gpt(diy_gpt_model_based_gpt2_config, gpt2_params)
# 获取设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diy_gpt_model_based_gpt2_config.to(device)

# 获取分词器
tokenizer = tiktoken.get_encoding("gpt2")

# 传入使用gpt2模型参数的大模型到文本生成函数， 生成新文本
torch.manual_seed(123)
token_ids = based_temperature_topk_generate_text_simple(
    gpt_model=diy_gpt_model_based_gpt2_config,
    index_array=text_to_tokens_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
)
print("生成的新文本 = ", token_ids_to_text(token_ids, tokenizer))
