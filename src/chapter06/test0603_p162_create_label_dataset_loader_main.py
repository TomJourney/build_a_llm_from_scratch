from pathlib import Path

import tiktoken
import torch
from torch.utils.data import DataLoader

from src.chapter06.test0603_p160_spam_dataset_module import DiySpamDataset

# 获取tiktoken中的gpt2分词器
gpt2_tokenizer = tiktoken.get_encoding("gpt2")

# 创建训练集
train_dataset = DiySpamDataset(
    csv_file=Path("dataset") / "train.csv",
    max_length=None,
    tokenizer=gpt2_tokenizer
)
# 创建验证集
validate_dataset = DiySpamDataset(
    csv_file=Path("dataset") / "validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=gpt2_tokenizer
)
# 创建测试集
test_dataset = DiySpamDataset(
    csv_file=Path("dataset") / "test.csv",
    max_length=train_dataset.max_length,
    tokenizer=gpt2_tokenizer
)

# 创建标签数据加载器
print("\n\n=== 创建标签数据加载器")
num_workers = 0
batch_size = 8
torch.manual_seed(123)

# 创建训练集数据加载器
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)
# 创建验证集数据加载器
validate_loader = DataLoader(
    dataset=validate_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=False
)
# 创建测试集数据加载器
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=False
)

print("\n\n=== 打印数据加载器的输入批次与目标批次")
for input_batch, target_batch in train_loader:
    pass
print("input_batch.shape = ", input_batch.shape)
print("target_batch.shape = ", target_batch.shape)
# input_batch.shape =  torch.Size([8, 120])
# target_batch.shape =  torch.Size([8])

print("\n\n===打印每个数据集的总批次数")
print("训练集批次数量 = ", len(train_loader))
print("验证集批次数量 = ", len(validate_loader))
print("测试集批次数量 = ", len(test_loader))
# 训练集批次数量 =  130
# 验证集批次数量 =  19
# 测试集批次数量 =  38
