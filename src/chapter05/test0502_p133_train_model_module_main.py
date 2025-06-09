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
# 创建自定义gpt模型实例
diy_gpt_model = DiyGPTModel(GPT_CONFIG_124M)
# 获取设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diy_gpt_model.to(device)

# 使用AdamW优化器和train_model_simple函数，对模型实例diy_gpt_model进行10轮训练
# .parameters()方法返回模型的所有可训练权重参数
optimizer = torch.optim.AdamW(
    diy_gpt_model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 10

# 训练模型
train_losses, test_losses, tokens_seen = train_model_simple(
    diy_gpt_model, train_data_loader, test_data_loader, optimizer, device, num_epochs, eval_frequency=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)
# Ep 1 (step 000000) :train_loss = 9.740, test_loss = 10.112
# Ep 1 (step 000005) :train_loss = 8.000, test_loss = 8.506
# decoded_text =  Every effort moves you,,,,,,...........,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
# Ep 2 (step 000010) :train_loss = 6.736, test_loss = 7.205
# Ep 2 (step 000015) :train_loss = 6.030, test_loss = 6.752
# decoded_text =  Every effort moves you.
# Ep 3 (step 000020) :train_loss = 5.533, test_loss = 6.569
# Ep 3 (step 000025) :train_loss = 5.650, test_loss = 6.600
# decoded_text =  Every effort moves you.
# Ep 4 (step 000030) :train_loss = 5.404, test_loss = 6.535
# Ep 4 (step 000035) :train_loss = 5.087, test_loss = 6.441
# decoded_text =  Every effort moves you.
# Ep 5 (step 000040) :train_loss = 4.627, test_loss = 6.431
# decoded_text =  Every effort moves you.
# Ep 6 (step 000045) :train_loss = 3.983, test_loss = 6.328
# Ep 6 (step 000050) :train_loss = 3.694, test_loss = 6.323
# decoded_text =  Every effort moves you know one of the picture--as of his own the fact with a little a little to have to see--and, and had been to me to have of his glory, and he had been his own a, and I had been he had been
# Ep 7 (step 000055) :train_loss = 3.181, test_loss = 6.330
# Ep 7 (step 000060) :train_loss = 2.893, test_loss = 6.181
# decoded_text =  Every effort moves you know the picture to the fact of the picture--I had the fact of the donkey, I had been--I have to the fact, in the picture--as that he had the donkey--the he had been the end of the picture.
# Ep 8 (step 000065) :train_loss = 2.440, test_loss = 6.329
# Ep 8 (step 000070) :train_loss = 2.139, test_loss = 6.173
# decoded_text =  Every effort moves you know the picture to the fact of the picture--I had the picture. "I was no great, the fact, and that, and I was his pictures--as he had been his painting, the fact, and his eyes.
# Ep 9 (step 000075) :train_loss = 1.712, test_loss = 6.211
# Ep 9 (step 000080) :train_loss = 1.509, test_loss = 6.201
# decoded_text =  Every effort moves you know," was not that my hostess was "interesting": on that point I could have given Miss Croft the fact of the last Mrs.           "I had the; and he had the his
# Ep 10 (step 000085) :train_loss = 1.153, test_loss = 6.271
# decoded_text =  Every effort moves you know," was one of the ax.  "I had the last word.    "I didn't about the you. "I was _not_ his pictures--the him up his ease--because he didn't want

