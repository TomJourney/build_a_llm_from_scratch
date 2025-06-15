import tiktoken
import torch

from src.chapter04.test0406_p107_gpt_model_module import DiyGPTModel
from src.chapter05.gpt_download import download_and_load_gpt2
from src.chapter05.test0505_p148_load_gpt2_params_to_diy_gpt_model_module import load_weights_into_gpt

# 【1】模型配置信息
CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"
# 基本配置，包括词汇表大小， 上下文长度， dropout率-丢弃率， 查询-键-值的偏置
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}
# 模型参数配置
# 字典保存不同模型尺寸的GPT模型参数
gpt2_model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (744M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25}
}
BASE_CONFIG.update(gpt2_model_configs[CHOOSE_MODEL])

# 加载预训练模型
pretrain_model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
print("pretrain_model_size = ", pretrain_model_size)  # 124M

# 获取gpt2模型的架构设置与权重参数
settings, params = download_and_load_gpt2(model_size=pretrain_model_size, models_dir="../chapter05/gpt2",
                                          is_download=False)
# 创建GPT模型实例
diy_gpt_model = DiyGPTModel(BASE_CONFIG)
# 把gpt2的参数加载到GPT模型实例
load_weights_into_gpt(diy_gpt_model, params)
# 把模型切换为推断模式 ，这会禁用模型的dropout层
diy_gpt_model.eval()

# 【1】 冻结模型，即将所有层设置为不可训练
for param in diy_gpt_model.parameters():
    param.requires_grad = False

# 【2】添加分类层， 替换输出层
torch.manual_seed(123)
num_classes = 2
print("BASE_CONFIG[\"emb_dim\"] = ", BASE_CONFIG["emb_dim"])
diy_gpt_model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],
    out_features=num_classes
)

# 【3】 为了使最终层归一化和最后一个Transformer块可训练， 本文把它们各自的requires_grad设置为True
for param in diy_gpt_model.transformer_blocks[-1].parameters():
    param.requires_grad = True
for param in diy_gpt_model.final_norm.parameters():
    param.requires_grad = True

# 获取tiktoken中的gpt2分词器
gpt2_tokenizer = tiktoken.get_encoding("gpt2")

# 像之前一样使用输出层被替换的模型
inputs = gpt2_tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)
print("inputs = ", inputs)
print("inputs.shape = ", inputs.shape)
# inputs =  tensor([[5211,  345,  423,  640]])
# inputs.shape =  torch.Size([1, 4])

# 把编码后的词元id传递给模型
with torch.no_grad():
    outputs = diy_gpt_model(inputs)
print("outputs = ", outputs)
print("outputs.shape = ", outputs.shape)
# outputs =  tensor([[[-1.4767,  5.5671],
#          [-2.4575,  5.3162],
#          [-1.0670,  4.5302],
#          [-2.3774,  5.1335]]])
# outputs.shape =  torch.Size([1, 4, 2])

# 输出张量中的最后一个词元
print("最后一个词元 = ", outputs[:, -1, :])
# 最后一个词元 =  tensor([[-1.9703,  4.2094]])

# 计算最高概率的位置
logits = outputs[:, -1, :]
label = torch.argmax(logits)
print("class label = ", label.item())
# class label =  1
