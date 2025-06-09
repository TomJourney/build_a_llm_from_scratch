from pathlib import Path

import tiktoken
import torch

from src.chapter02.test0206_p35_dataloader import create_data_loader_v1
from src.chapter04.test0406_p107_gpt_model_module import DiyGPTModel
from src.chapter05.test0502_p133_train_model_module import train_model_simple
from src.utils import BusiIoUtils

# 读取verdict小说
print("\n\n=== 读取verdict小说")
with open(Path(BusiIoUtils.get_root_dir(), "..", "file", "the-verdict.txt")) as f:
    raw_text = f.read()

# 把数据分为训练集与测试集，并使用第2章的数据加载器来准备大模型训练所需的批次数据
train_ratio = 0.9
split_index = int(train_ratio * len(raw_text))
train_data = raw_text[:split_index]
test_data = raw_text[split_index:]

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

# 使用AdamW优化器和train_model_simple函数对 DiyGPTModel进行10次训练

# 创建数据加载器
torch.manual_seed(123)
train_data_loader = create_data_loader_v1(train_data,
                                          batch_size=2,
                                          max_length=GPT_CONFIG_124M["context_length"],
                                          stride=GPT_CONFIG_124M["context_length"],
                                          drop_last=False,
                                          shuffle=False,
                                          num_workers=0)
test_data_loader = create_data_loader_v1(test_data,
                                         batch_size=2,
                                         max_length=GPT_CONFIG_124M["context_length"],
                                         stride=GPT_CONFIG_124M["context_length"],
                                         drop_last=False,
                                         shuffle=False,
                                         num_workers=0)

# 实例化gpt模型（使用自定义GPT模型）
torch.manual_seed(123)
# 获取分词器
tokenizer = tiktoken.get_encoding("gpt2")
diy_gpt_model = DiyGPTModel(GPT_CONFIG_124M)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diy_gpt_model.to(device)
optimizer = torch.optim.AdamW(
    diy_gpt_model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 10
train_losses, test_losses, tokens_seen = train_model_simple(
    diy_gpt_model, train_data_loader, test_data_loader, optimizer, device, num_epochs, eval_frequency=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)
