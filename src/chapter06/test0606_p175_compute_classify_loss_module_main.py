from pathlib import Path

import tiktoken
import torch
from torch.utils.data import DataLoader

from src.chapter04.test0406_p107_gpt_model_module import DiyGPTModel
from src.chapter05.gpt_download import download_and_load_gpt2
from src.chapter05.test0505_p148_load_gpt2_params_to_diy_gpt_model_module import load_weights_into_gpt
from src.chapter06.test0603_p160_spam_dataset_module import DiySpamDataset
from src.chapter06.test0606_p175_compute_classify_loss_module import compute_classify_loss_loader

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

# 2 计算分类准确率
# 获取tiktoken中的gpt2分词器
gpt2_tokenizer = tiktoken.get_encoding("gpt2")
num_workers = 0
batch_size = 8
torch.manual_seed(123)

# 2.1 计算训练集分类正确率
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

# 2.2 计算验证集分类正确率
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

# 2.3 计算测试集分类正确率
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

# 整合计算训练集，验证集，测试集分类正确率
with torch.no_grad():
    train_loss = compute_classify_loss_loader(train_loader, diy_gpt_model, device, num_batches=5)
    validate_loss = compute_classify_loss_loader(validate_loader, diy_gpt_model, device, num_batches=5)
    test_loss = compute_classify_loss_loader(test_loader, diy_gpt_model, device, num_batches=5)

print(f"训练集分类损失 = {train_loss:.3f}")
print(f"验证集分类损失 = {validate_loss:.3f}")
print(f"测试集分类损失 = {test_loss:.3f}")
# 训练集分类损失 = 7.973
# 验证集分类损失 = 8.861
# 测试集分类损失 = 8.393