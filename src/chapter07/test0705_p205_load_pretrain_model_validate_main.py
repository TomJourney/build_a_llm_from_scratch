import json

import tiktoken
import torch
from torch.utils.data import DataLoader

from src.chapter04.test0406_p107_gpt_model_module import DiyGPTModel
from src.chapter05.gpt_download import download_and_load_gpt2
from src.chapter05.test0501_p119_text_to_token_transfer_util_module import text_to_tokens_ids, token_ids_to_text
from src.chapter05.test0503_p142_modify_text_generate_function import based_temperature_topk_generate_text_simple
from src.chapter05.test0505_p148_load_gpt2_params_to_diy_gpt_model_module import load_weights_into_gpt
from src.chapter07.test0702_p189_format_input_to_alpaca_module import format_input_to_alpaca
from src.chapter07.test0703_p193_instruction_dataset_module import InstructionDataset
from src.chapter07.test0703_p194_custom_agg_module import custom_agg_function_v2

# 读取数据集
with open('./dataset/instruction-data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

print("\n=== 划分数据集： 训练集85%， 测试集10%， 验证集5%")
train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
validate_portion = int(len(data) * 0.05)

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
validate_data = data[train_portion + test_portion:]

print("\n\n=== 初始化用于指令微调的数据加载器")
num_workers = 0
batch_size = 8
torch.manual_seed(123)

# 获取tiktoken中的gpt2分词器
gpt2_tokenizer = tiktoken.get_encoding("gpt2")

print("\n=== 初始化训练数据加载器")
train_dataset = InstructionDataset(train_data, gpt2_tokenizer)
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              collate_fn=custom_agg_function_v2,
                              shuffle=False,
                              drop_last=False,
                              num_workers=num_workers)

print("\n=== 1 初始化训练集数据加载器")
train_dataset = InstructionDataset(train_data, gpt2_tokenizer)
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              collate_fn=custom_agg_function_v2,
                              shuffle=False,
                              drop_last=False,
                              num_workers=num_workers)

print("\n=== 2 初始化验证集数据加载器")
validate_dataset = InstructionDataset(validate_data, gpt2_tokenizer)
validate_dataloader = DataLoader(validate_dataset,
                                 batch_size=batch_size,
                                 collate_fn=custom_agg_function_v2,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=num_workers)

print("\n=== 3 初始化测试集数据加载器")
test_dataset = InstructionDataset(test_data, gpt2_tokenizer)
test_dataloader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             collate_fn=custom_agg_function_v2,
                             shuffle=False,
                             drop_last=False,
                             num_workers=num_workers)

print("\n=== 【测试案例1】加载预训练模型")
# 测试用例-计算分类准确率
# 【1】模型配置信息
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
# 选择参数量为3.55亿的模型
CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(gpt2_model_configs[CHOOSE_MODEL])

# 解析模型的参数大小
pretrain_model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
print("pretrain_model_size = ", pretrain_model_size)  # 355M

# 下载模型
settings, params = download_and_load_gpt2(model_size=pretrain_model_size, models_dir="gpt2", is_download=False)

# 创建大模型实例，加载权重到模型实例
gpt2_355_model = DiyGPTModel(BASE_CONFIG)
load_weights_into_gpt(gpt2_355_model, params)
# 设置大模型为评估模式
gpt2_355_model.eval()

print("\n===【测试案例2】基于验证集评估大模型")
torch.manual_seed(123)
input_text = format_input_to_alpaca(validate_data[0])
print("input_text = ", input_text)

print("\n=== 使用generate函数-based_temperature_topk_generate_text_simple生成模型的回复")
token_ids = based_temperature_topk_generate_text_simple(
    gpt_model=gpt2_355_model,
    index_array=text_to_tokens_ids(input_text, gpt2_tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256
)
generated_text = token_ids_to_text(token_ids, gpt2_tokenizer)

# based_temperature_topk_generate_text_simple函数返回的是拼接在一起的输入与输出文本
# 而本测试用例仅关注模型生成的回复
print("\n=== 获取模型的回复")
response_text = generated_text[len(input_text):].strip()
print("模型回复 = ", response_text)
