import json
import time

import tiktoken
import torch
from torch.utils.data import DataLoader

from src.chapter04.test0406_p107_gpt_model_module import DiyGPTModel
from src.chapter05.gpt_download import download_and_load_gpt2
from src.chapter05.test0502_p133_train_model_module import train_model_simple
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
# 选择参数量为1.2亿的模型
CHOOSE_MODEL = "gpt2-small (124M)"
# 选择参数量为3.55亿的模型
# CHOOSE_MODEL = "gpt2-medium (355M)"

BASE_CONFIG.update(gpt2_model_configs[CHOOSE_MODEL])

# 解析模型的参数大小
pretrain_model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
print("pretrain_model_size = ", pretrain_model_size)  # 355M

# 下载模型
settings, params = download_and_load_gpt2(model_size=pretrain_model_size, models_dir="../chapter05/gpt2", is_download=False)

# 创建大模型实例，加载权重到模型实例
gpt2_model = DiyGPTModel(BASE_CONFIG)
load_weights_into_gpt(gpt2_model, params)
# 设置大模型为评估模式
gpt2_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练大模型，训练过程包括初始化优化器，设定训练轮数，定义评估的频率和起始上下文
# 测试案例-对预训练的大模型进行指令微调

start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(
    gpt2_model.parameters(), lr=0.00005, weight_decay=0.1
)
# 训练轮次
num_epochs = 1
# 训练模型
train_loss, validate_loss, tokens_seen = train_model_simple(
    gpt2_model,
    train_dataloader,
    validate_dataloader,
    optimizer,
    device,
    num_epochs == num_epochs,
    eval_frequency=5,
    eval_iter=4,
    start_context=format_input_to_alpaca(validate_data[0]),
    tokenizer=gpt2_tokenizer
)
end_time = time.time()
execution_time_minute = (end_time-start_time)/60
print(f"training completed in {execution_time_minute:.2f} minutes.")
# pretrain_model_size =  124M
# Ep 1 (step 000000) :train_loss = 7.778, validate_loss = 7.870
# Ep 1 (step 000005) :train_loss = 6.129, validate_loss = 6.339
# Ep 1 (step 000010) :train_loss = 4.984, validate_loss = 5.279
# Ep 1 (step 000015) :train_loss = 4.260, validate_loss = 4.626
# Ep 1 (step 000020) :train_loss = 3.822, validate_loss = 4.237
# Ep 1 (step 000025) :train_loss = 3.523, validate_loss = 3.940
# Ep 1 (step 000030) :train_loss = 3.319, validate_loss = 3.735
# Ep 1 (step 000035) :train_loss = 3.147, validate_loss = 3.581
# Ep 1 (step 000040) :train_loss = 3.027, validate_loss = 3.472
# Ep 1 (step 000045) :train_loss = 2.914, validate_loss = 3.362
# Ep 1 (step 000050) :train_loss = 2.842, validate_loss = 3.304
# Ep 1 (step 000055) :train_loss = 2.778, validate_loss = 3.220
# Ep 1 (step 000060) :train_loss = 2.723, validate_loss = 3.174
# Ep 1 (step 000065) :train_loss = 2.666, validate_loss = 3.125
# Ep 1 (step 000070) :train_loss = 2.622, validate_loss = 3.087
# Ep 1 (step 000075) :train_loss = 2.576, validate_loss = 3.045
# Ep 1 (step 000080) :train_loss = 2.536, validate_loss = 3.021
# Ep 1 (step 000085) :train_loss = 2.501, validate_loss = 3.001
# Ep 1 (step 000090) :train_loss = 2.487, validate_loss = 3.003
# Ep 1 (step 000095) :train_loss = 2.458, validate_loss = 2.964
# Ep 1 (step 000100) :train_loss = 2.440, validate_loss = 2.932
# Ep 1 (step 000105) :train_loss = 2.411, validate_loss = 2.907
# Ep 1 (step 000110) :train_loss = 2.393, validate_loss = 2.893
# Ep 1 (step 000115) :train_loss = 2.369, validate_loss = 2.871
# training completed in 12.65 minutes.
