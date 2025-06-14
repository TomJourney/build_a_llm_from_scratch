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
                                          drop_last=True,
                                          shuffle=True,
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
train_losses, validate_losses, tokens_seen = train_model_simple(
    diy_gpt_model, train_data_loader, test_data_loader, optimizer, device, num_epochs, eval_frequency=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)
# Ep 1 (step 000000) :train_loss = 9.781, validate_loss = 9.933
# Ep 1 (step 000005) :train_loss = 8.111, validate_loss = 8.339
# decoded_text =  Every effort moves you,,,,,,,,,,,,.
# Ep 2 (step 000010) :train_loss = 6.661, validate_loss = 7.048
# Ep 2 (step 000015) :train_loss = 5.961, validate_loss = 6.616
# decoded_text =  Every effort moves you, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and,, and, and,
# Ep 3 (step 000020) :train_loss = 5.726, validate_loss = 6.600
# Ep 3 (step 000025) :train_loss = 5.201, validate_loss = 6.348
# decoded_text =  Every effort moves you, and I had been.
# Ep 4 (step 000030) :train_loss = 4.417, validate_loss = 6.278
# Ep 4 (step 000035) :train_loss = 4.069, validate_loss = 6.226
# decoded_text =  Every effort moves you know the                          "I he had the donkey and I had the and I had the donkey and down the room, I had
# Ep 5 (step 000040) :train_loss = 3.732, validate_loss = 6.160
# decoded_text =  Every effort moves you know it was not that the picture--I had the fact by the last I had been--his, and in the            "Oh, and he said, and down the room, and in
# Ep 6 (step 000045) :train_loss = 2.850, validate_loss = 6.179
# Ep 6 (step 000050) :train_loss = 2.427, validate_loss = 6.141
# decoded_text =  Every effort moves you know," was one of the picture. The--I had a little of a little: "Yes, and in fact, and in the picture was, and I had been at my elbow and as his pictures, and down the room, I had
# Ep 7 (step 000055) :train_loss = 2.104, validate_loss = 6.134
# Ep 7 (step 000060) :train_loss = 1.882, validate_loss = 6.233
# decoded_text =  Every effort moves you know," was one of the picture for nothing--I told Mrs.  "I was no--as! The women had been, in the moment--as Jack himself, as once one had been the donkey, and were, and in his
# Ep 8 (step 000065) :train_loss = 1.320, validate_loss = 6.238
# Ep 8 (step 000070) :train_loss = 0.985, validate_loss = 6.242
# decoded_text =  Every effort moves you know," was one of the axioms he had been the tips of a self-confident moustache, I felt to see a smile behind his close grayish beard--as if he had the donkey. "strongest," as his
# Ep 9 (step 000075) :train_loss = 0.717, validate_loss = 6.293
# Ep 9 (step 000080) :train_loss = 0.541, validate_loss = 6.393
# decoded_text =  Every effort moves you?"  "Yes--quite insensible to the irony. She wanted him vindicated--and by me!"  He laughed again, and threw back the window-curtains, I had the donkey. "There were days when I
# Ep 10 (step 000085) :train_loss = 0.391, validate_loss = 6.452
# decoded_text =  Every effort moves you know," was one of the axioms he laid down across the Sevres and silver of an exquisitely appointed luncheon-table, when, on a later day, I had again run over from Monte Carlo; and Mrs. Gis

