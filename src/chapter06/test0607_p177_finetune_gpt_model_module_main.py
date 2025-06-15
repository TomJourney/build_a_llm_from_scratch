import time
from pathlib import Path

import tiktoken
import torch
from torch.utils.data import DataLoader

from src.chapter04.test0406_p107_gpt_model_module import DiyGPTModel
from src.chapter05.gpt_download import download_and_load_gpt2
from src.chapter05.test0505_p148_load_gpt2_params_to_diy_gpt_model_module import load_weights_into_gpt
from src.chapter06.test0603_p160_spam_dataset_module import DiySpamDataset
from src.chapter06.test0607_p177_finetune_gpt_model_module import train_classifier_simple
from src.chapter06.test0607_p179_plot_classify_loss_module import plot_values
from src.chapter06.test0606_p174_compute_classify_accuracy_module import compute_accuracy_loader

# 测试用例-计算分类准确率
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diy_gpt_model.to(device)

# 获取tiktoken中的gpt2分词器
gpt2_tokenizer = tiktoken.get_encoding("gpt2")
num_workers = 0
batch_size = 8
torch.manual_seed(123)

# 创建训练集加载器
train_dataset = DiySpamDataset(
    csv_file=Path("dataset") / "train.csv",
    max_length=None,
    tokenizer=gpt2_tokenizer
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)
# 创建验证集加载器
validate_dataset = DiySpamDataset(
    csv_file=Path("dataset") / "validation.csv",
    max_length=None,
    tokenizer=gpt2_tokenizer
)
validate_loader = DataLoader(
    dataset=validate_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=False
)
# 创建测试集加载器
test_dataset = DiySpamDataset(
    csv_file=Path("dataset") / "test.csv",
    max_length=None,
    tokenizer=gpt2_tokenizer
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=False
)

# 训练开始： 使用微调模型函数进行分类训练
print("\n\n=== 训练开始： 使用微调模型函数进行分类训练 ")
start_time = time.time()
optimizer = torch.optim.AdamW(diy_gpt_model.parameters(), lr=5e-5, weight_decay=0.1)
# 训练轮次=5
num_epochs = 5
# 执行train_classifier_simple-函数进行训练
train_losses, validate_losses, train_accurate_array, validate_accurate_array, examples_seen = train_classifier_simple(
    diy_gpt_model, train_loader, validate_loader, optimizer, device, num_epochs=num_epochs, eval_freq=50, eval_iter=5
)
end_time = time.time()
exec_minute_cost = (end_time - start_time) / 60
print("使用微调模型函数进行分类训练， 耗时（分钟）=", exec_minute_cost)

# === 训练开始： 使用微调模型函数进行分类训练
# Ep 1 step 000000: train loss = 6.484, validate loss = 5.762
# Ep 1 step 000050: train loss = 0.375, validate loss = 0.376
# Ep 1 step 000100: train loss = 0.212, validate loss = 0.409
# train_accuracy = 87.50%
# validate_accuracy = 87.50%
# Ep 2 step 000150: train loss = 0.361, validate loss = 0.474
# Ep 2 step 000200: train loss = 0.427, validate loss = 0.577
# Ep 2 step 000250: train loss = 0.513, validate loss = 0.587
# train_accuracy = 50.00%
# validate_accuracy = 87.50%
# Ep 3 step 000300: train loss = 0.144, validate loss = 0.380
# Ep 3 step 000350: train loss = 0.146, validate loss = 0.252
# train_accuracy = 75.00%
# validate_accuracy = 62.50%
# Ep 4 step 000400: train loss = 0.081, validate loss = 0.205
# Ep 4 step 000450: train loss = 0.069, validate loss = 0.265
# Ep 4 step 000500: train loss = 0.031, validate loss = 0.194
# train_accuracy = 100.00%
# validate_accuracy = 100.00%
# Ep 5 step 000550: train loss = 0.055, validate loss = 0.133
# Ep 5 step 000600: train loss = 0.009, validate loss = 0.049
# train_accuracy = 100.00%
# validate_accuracy = 87.50%
# 使用微调模型函数进行分类训练， 耗时（分钟）= 81.96330119371414

# 绘制分类损失曲线
print("\n===绘制分类损失曲线")
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_sensor = torch.linspace(0, examples_seen, len(train_losses))
plot_values(epochs_tensor, examples_seen_sensor, train_losses, validate_losses)

# 绘制分类准确率曲线
print("\n===绘制分类准确率曲线 ")
epochs_tensor = torch.linspace(0, num_epochs, len(train_accurate_array))
examples_seen_sensor = torch.linspace(0, examples_seen, len(train_accurate_array))
plot_values(epochs_tensor, examples_seen_sensor, train_accurate_array, validate_accurate_array, label="accuracy")

# 计算整个数据集在训练集，验证集和测试集上的性能指标
train_accuracy = compute_accuracy_loader(train_loader, diy_gpt_model, device)
validate_accuracy = compute_accuracy_loader(validate_loader, diy_gpt_model, device)
test_accuracy = compute_accuracy_loader(test_loader, diy_gpt_model, device)
print(f"训练集分类准确率 = {train_accuracy * 100:.2f}%")
print(f"验证集分类准确率 = {validate_accuracy * 100:.2f}%")
print(f"测试集分类准确率 = {test_accuracy * 100:.2f}%")

