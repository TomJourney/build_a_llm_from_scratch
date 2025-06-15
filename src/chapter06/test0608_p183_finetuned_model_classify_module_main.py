from pathlib import Path

import tiktoken
import torch

from src.chapter04.test0406_p107_gpt_model_module import DiyGPTModel
from src.chapter05.gpt_download import download_and_load_gpt2
from src.chapter05.test0505_p148_load_gpt2_params_to_diy_gpt_model_module import load_weights_into_gpt
from src.chapter06.test0603_p160_spam_dataset_module import DiySpamDataset
from src.chapter06.test0608_p183_finetuned_model_classify_module import classify_review

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

# 使用classify_review函数分类垃圾消息
print("\n=== 使用classify_review函数分类垃圾消息（texxt_1）")
text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)
print(classify_review(text_1, diy_gpt_model, gpt2_tokenizer, device, max_length=train_dataset.max_length))
# spam (垃圾消息)

print("\n=== 使用classify_review函数分类垃圾消息（texxt_2）")
text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)
print(classify_review(text_2, diy_gpt_model, gpt2_tokenizer, device, max_length=train_dataset.max_length))
# not spam (非垃圾消息)

print("\n\n=== 保存训练好的分类模型")
torch.save(diy_gpt_model.state_dict(), "review_classifier.pth")

print("\n\n=== 加载保存好的分类模型")
model_state_dict = torch.load("review_classifier.pth", map_location=device)
diy_gpt_model.load_state_dict(model_state_dict)




