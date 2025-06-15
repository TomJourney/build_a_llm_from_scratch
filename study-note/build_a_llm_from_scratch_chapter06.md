[TOC]

# 【README】

本文总结自<font color="#ff0000">《从零构建大模型》</font>，非常棒的一本书，墙裂推荐； 

本文通过在特定目标任务（如文本分类）上<font color=red>微调大模型</font>，来实践之前的学习成果；

微调大模型有两种方式，如图6-1所示：

- 用于分类的微调；
- 用于执行指令的微调；

本文代码参见： [https://github.com/TomJourney/build_a_llm_from_scratch](https://github.com/TomJourney/build_a_llm_from_scratch)

<br>

【图6-1】 构建大模型的3个主要阶段：1-实现一个类GPT大模型架构；2-把预训练模型的权重加载到大模型架构中；3-微调预训练的大模型来给文本分类；

![image-20250615090742856](./pic/06/0601.png)



---

# 【1】对大模型进行分类微调的3个阶段

微调语言模型最常见的方法： 指令微调、分类微调；

- 指令微调：如判断文本是否为垃圾消息，句子翻译等；
- 指令微调：提升了模型基于特定用户指令理解和生成响应的能力。指令微调最适合处理需要应对多种任务的模型，这些任务依赖于复杂的用户指令。

<br>

---

## 【1.1】对大模型进行分类微调的3个阶段

对大模型进行分类微调的3个阶段：包括准备数据集，模型设置，模型微调和应用；如图6-4所示。

1. 准备数据集：
   1. 下载数据集；
   2. 数据集预处理；
   3. 创建数据加载器；
2. 模型设置：
   1. 模型初始化；；
   2. 加载预训练权重；
   3. 修改模型以便微调；
   4. 实现评估工具； 
3. 模型微调与应用：
   1. 微调模型；
   2. 评估微调后的模型；
   3. 在新数据上应用模型； 



![image-20250615091451980](./pic/06/0604.png)

---

# 【2】准备数据集

## 【2.1】下载数据集

【test0601_p156_download_unzip_dataset_module.py】下载数据集

```python
import os
import urllib.request
import zipfile
from pathlib import Path

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. skipping download and extraction")
        return

    # 下载文件
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    # 解压文件
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    # 添加.tsv文件扩展名
    os.rename(original_file_path, data_file_path)
    print(f"file downloaded and saved as {data_file_path}")
```

【test0601_p156_download_unzip_dataset_module_main.py】测试案例-下载数据集并查看数据分布

```python
from pathlib import Path

import pandas as pd

# 下载并解压数据集
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
print("下载并解压数据集完成")

# 加载数据集
data_file_path = Path("dataset") / "SMSSpamCollection.tsv"
df = pd.read_csv(
    data_file_path, sep="\t", header=None, names=["Label", "Text"]
)

print("\n\n===查看数据分布")
print(f"数据分布 = \n", df["Label"].value_counts())
#  Label
# ham     4825
# spam     747
# Name: count, dtype: int64
```

<br>

---

### 【2.1.1】创建平衡数据集 

为简单起见，本文选择747的spam垃圾文本数据集，并创建包含747的非垃圾文本数据集，垃圾文本与非垃圾文本的数据量相等，称为平衡数据集，如下。

平衡数据集定义： 各类别的数量相同的数据集；

【test0602_p157_create_balance_dataset_module.py】创建平衡数据集模块

```python
import pandas as pd


def create_balanced_dataset(df):
    # 统计垃圾消息的样本数量
    num_spam = df[df["Label"] == "spam"].shape[0]

    # 随机采样非垃圾消息，使其数量与垃圾消息一致
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state=123
    )

    # 将垃圾消息与采样后的垃圾消息组合，构成平衡数据集
    balanced_df = pd.concat([
        ham_subset, df[df["Label"] == "spam"]
    ])
    return balanced_df
```

【test0602_p157_create_balance_dataset_module_main.py】测试案例-创建平衡数据集模块

```python
from pathlib import Path

import pandas as pd

from src.chapter06.test0601_p156_download_unzip_dataset_module import download_and_unzip_spam_data
from src.chapter06.test0602_p157_create_balance_dataset_module import create_balanced_dataset

# 下载并解压数据集
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
print("下载并解压数据集完成")

# 加载数据集
data_file_path = Path("dataset") / "SMSSpamCollection.tsv"
df = pd.read_csv(
    data_file_path, sep="\t", header=None, names=["Label", "Text"]
)

print("\n\n===查看数据分布")
print("数据分布 = \n", df["Label"].value_counts())
#  Label
# ham     4825
# spam     747
# Name: count, dtype: int64

print("\n\n=== 创建平衡数据集")
balanced_df = create_balanced_dataset(df)
print("balanced_df = \n", balanced_df["Label"].value_counts())
#  Label
# ham     747
# spam    747
# Name: count, dtype: int64
```

---

## 【2.2】划分数据集

把数据集划分为3部分，包括训练数据集-70%， 验证数据集-10%， 测试数据集=20% 

【test0602_p158_split_dataset_module.py】划分数据集模块

```python
# train_fraction 训练集比例
# validation_fraction 验证集比例
def random_split(df, train_fraction, validation_fraction):
    # 打乱整个 Dataframe
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # 计算拆分索引
    train_end = int(len(df) * train_fraction)
    validation_end = train_end + int(len(df) * validation_fraction)

    # 拆分 Dataframe，包括训练集，验证集， 测试集
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df
```

【test0602_p158_split_dataset_module_main.py】测试案例-划分数据集模块

```python
from pathlib import Path

import pandas as pd

from src.chapter06.test0602_p157_create_balance_dataset_module import create_balanced_dataset
from src.chapter06.test0602_p158_split_dataset_module import random_split

# 下载并解压数据集
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
print("下载并解压数据集完成")

# 加载数据集
data_file_path = Path("dataset") / "SMSSpamCollection.tsv"
df = pd.read_csv(
    data_file_path, sep="\t", header=None, names=["Label", "Text"]
)

print("\n\n=== 步骤1： 查看数据分布")
print("数据分布 = \n", df["Label"].value_counts())
#  Label
# ham     4825
# spam     747
# Name: count, dtype: int64

print("\n\n=== 步骤2： 创建平衡数据集")
balanced_df = create_balanced_dataset(df)
print("balanced_df = \n", balanced_df["Label"].value_counts())
#  Label
# ham     747
# spam    747
# Name: count, dtype: int64

print("\n\n=== 步骤2-1： 把ham与spam分别转换为标签0和1")
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

print("\n\n=== 步骤3： 划分数据集，训练集70%， 验证集10%， 测试集20%； 其中总数量=747*2=1494")
train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
print("把数据集保存为csv，以便重用")
train_df.to_csv(Path("dataset") / "train.csv", index=None)
validation_df.to_csv(Path("dataset") / "validation.csv", index=None)
test_df.to_csv(Path("dataset") / "test.csv", index=None)

print("\n\n===统计各数据集的数据量")
print("训练集数据量 = ", pd.read_csv(Path("dataset")/"train.csv").shape)
print("验证集数据量 = ", pd.read_csv(Path("dataset")/"validation.csv").shape)
print("测试集数据量 = ", pd.read_csv(Path("dataset")/"test.csv").shape)
# ===统计各数据集的数据量
# 训练集数据量 =  (1045, 2)
# 验证集数据量 =  (149, 2)
# 测试集数据量 =  (300, 2)
```

<br>

---

# 【3】创建数据加载器 

为实现批处理，本文把所有消息都填充到最长消息的长度，需要向较短消息添加填充词元<|endoftext|>，具体实现是把文本<|endoftext|>对应的词元id添加到编码的文本消息中。

50256是填充词元<|endoftext|>的词元id。可以使用tiktoken包中的gpt-2分词器来核对填充词元id是否为50256。

```python
import tiktoken

# 获取tiktoken中的gpt2分词器
gpt2_tokenizer = tiktoken.get_encoding("gpt2")
# 把<|endoftext|>作为填充词元，词元<|endoftext|>的词元id等于50256
print(gpt2_tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
# [50256]
```

## 【3.1】创建自定义垃圾文本数据集类

【test0603_p160_spam_dataset_module.py】创建自定义垃圾文本数据集模块 

```python
import pandas as pd
from sympy.printing.pytorch import torch
from torch.utils.data import Dataset


# 垃圾邮件数据集类-DiySpamDataset
# 垃圾邮件数据集类处理几个关键任务： 1-把文本消息编码为词元序列，2-识别训练数据集中最长的序列，
# 3-确保所有其他序列都使用填充词元进行填充，以匹配最长序列的长度
class DiySpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        # 文本分词
        self.encoded_text_array = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

            # 如果序列长度超过 max_length，则进行截断
            self.encoded_text_array = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_text_array
            ]

        # 填充到最长序列的长度
        self.encoded_text_array = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_text_array
        ]

    def __getitem__(self, index):
        encoded = self.encoded_text_array[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    # 识别数据集中数据序列的最大长度
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_text_array:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
```

【test0603_p160_spam_dataset_module_main.py】测试案例-创建自定义垃圾文本数据集模块 

```python
from pathlib import Path

import tiktoken

from src.chapter06.test0603_p160_spam_dataset_module import DiySpamDataset

# 获取tiktoken中的gpt2分词器
gpt2_tokenizer = tiktoken.get_encoding("gpt2")
# 把<|endoftext|>作为填充词元，词元<|endoftext|>的词元id等于50256
print(gpt2_tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
# [50256]

# 创建训练集
train_dataset = DiySpamDataset(
    csv_file=Path("dataset") / "train.csv",
    max_length=None,
    tokenizer=gpt2_tokenizer
)
# 最长序列长度
print("train_dataset.max_length = ", train_dataset.max_length)
# train_dataset.max_length =  120

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
```

---

## 【3.2】创建数据加载器 

与创建文本数据加载器类似，可以创建标签数据加载器，即目标是类别标签，而不是文本中的下一个词元； 

选择批次大小为8，则每个批次将包含8个样本，每个样本120个词元（或词元id），对应8个类别标签。如图6-7所示。

![image-20250615171553124](./pic/06/0607.png) 

---

### 【3.2.1】标签数据加载器代码实现

【test0603_p162_create_label_dataset_loader_main.py】测试案例-创建标签数据加载器 

```python
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
```

---

<br>

# 【4】初始化带有预训练权重的模型 

初始化预训练模型：对垃圾消息进行分类微调的第一步，是要初始化预训练模型，如图6-8所示；  

【test0604_p164_initialize_pretrained_model_main.py】测试案例-初始化带有预训练权重的模型 

```python
import tiktoken

from src.chapter04.test0406_p107_gpt_model_module import DiyGPTModel
from src.chapter05.gpt_download import download_and_load_gpt2
from src.chapter05.test0501_p119_text_to_token_transfer_util_module import text_to_tokens_ids, token_ids_to_text
from src.chapter05.test0503_p142_modify_text_generate_function import based_temperature_topk_generate_text_simple
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

# 【2】加载预训练模型
pretrain_model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
print("pretrain_model_size = ", pretrain_model_size) # 124M

# 获取gpt2模型的架构设置与权重参数
settings, params = download_and_load_gpt2(model_size=pretrain_model_size, models_dir="../chapter05/gpt2", is_download=False)
# 创建GPT模型实例
diy_gpt_model = DiyGPTModel(BASE_CONFIG)
# 把gpt2的参数加载到GPT模型实例
load_weights_into_gpt(diy_gpt_model, params)
# 把模型切换为推断模式 ，这会禁用模型的dropout层
diy_gpt_model.eval()

print("\n=== 【3】 使用文本生成工具函数， 确保模型生成连贯的文本")
# 获取tiktoken中的gpt2分词器
gpt2_tokenizer = tiktoken.get_encoding("gpt2")

text_1 = "Every effort moves you"
token_ids = based_temperature_topk_generate_text_simple(
    gpt_model=diy_gpt_model,
    index_array=text_to_tokens_ids(text_1, tokenizer=gpt2_tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, gpt2_tokenizer))

print("\n=== 【4】 把模型微调为垃圾消息分类器之前， 本文尝试输入指令，查看模型是否能够正确分类垃圾消息")
test_text = (
    "Is the following text 'spam' ? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)

test_token_ids = based_temperature_topk_generate_text_simple(
    gpt_model=diy_gpt_model,
    index_array=text_to_tokens_ids(test_text, tokenizer=gpt2_tokenizer),
    max_new_tokens=23,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(test_token_ids, gpt2_tokenizer))
```

<br>

---

# 【5】添加分类头 

本文的目标是预测文本消息是否为垃圾消息，输出标签是1-垃圾消息， 0-非垃圾消息。

所以需要把原始输出层（该输出层映射到一张包含50257个词元的词汇表），替换为一个较小的输出层，该输出层映射到两个类别，即0与1； 如6-9所示（本文使用了之前的模型，但替换了输出层）。

![image-20250615180128003](./pic/06/0609.png) 

---

## 【5.1】替换原始模型的输出层

本文接下来把原始模型输出层out_head 替换为新的输出层，并对其微调。

由于模型经过了预训练，无需微调所有层。只微调最后几层通常足以使新模型适应新任务。同时，仅微调少量层在计算上也更加高效。

【test0605_p168_add_classification_head_main.py】 测试案例-添加分类层 

```python
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

```

【代码解说】

之前的输出的形状为[1, 4, 50257]，而替换输出层后的输出为[1, 4, 2]。

本文的目标是微调此模型，使其返回一个类别标签，以指出输入是垃圾消息还是非垃圾消息；

因此，<font color=red>本文只需要关注最后一个词元（最后一行）即可，因为序列中的最后一个词元累计了最多的信息 </font>。 

```python
# 输出张量中的最后一个词元
print("最后一个词元 = ", outputs[:, -1, :])
# 最后一个词元 =  tensor([[-1.9703,  4.2094]]) 
```

---

# 【6】计算分类损失和准确率（实现模型评估函数）

模型评估函数（工具）：评估模型性能的函数，以评估模型在微调前、微调中和微调后的分类垃圾邮件的性能 ； 

## 【6.1】把模型输出转换为类别标签预测

自定义GPT模型的输出是概率，与之前下一个词元预测方法类似，本文计算最高概率的位置（下标）；

【test0605_p168_add_classification_head_main.py】 测试案例-新增分类头

```python
# 输出张量中的最后一个词元
print("最后一个词元 = ", outputs[:, -1, :])
# 最后一个词元 =  tensor([[-1.9703,  4.2094]])

# 计算最高概率的位置
logits = outputs[:, -1, :]
label = torch.argmax(logits)
print("class label = ", label.item())
# class label =  1
```

---

## 【6.2】计算分类准确率

【test0606_p174_compute_classify_accuracy_module.py】计算分类准确率

```python
import torch


# 计算分类准确率加载器
def compute_accuracy_loader(data_loader, diy_gpt_model, device, num_batches=None):
    diy_gpt_model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            # 最后一个输出词元的logits
            with torch.no_grad():
                logits = diy_gpt_model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (
                (predicted_labels == target_batch).sum().item()
            )
        else:
            break
        return correct_predictions / num_examples
```

【test0606_p174_compute_classify_accuracy_module_main.py】测试案例-计算分类准确率

```python
from pathlib import Path

import tiktoken
import torch
from torch.utils.data import DataLoader

from src.chapter04.test0406_p107_gpt_model_module import DiyGPTModel
from src.chapter05.gpt_download import download_and_load_gpt2
from src.chapter05.test0505_p148_load_gpt2_params_to_diy_gpt_model_module import load_weights_into_gpt
from src.chapter06.test0603_p160_spam_dataset_module import DiySpamDataset
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
train_accuracy = compute_accuracy_loader(
    train_loader, diy_gpt_model, device, num_batches=10
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
validate_accuracy = compute_accuracy_loader(
    validate_loader, diy_gpt_model, device, num_batches=10
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
test_accuracy = compute_accuracy_loader(
    test_loader, diy_gpt_model, device, num_batches=10
)
print(f"训练集分类准确率 = {train_accuracy * 100:.2f}%")
print(f"验证集分类准确率 = {validate_accuracy * 100:.2f}%")
print(f"测试集分类准确率 = {test_accuracy * 100:.2f}%")
```

【代码解说】

为了提高预测准确率，需要对模型进行微调； 

在微调前，需要定义损失函数；因为分类准确率不是一个可微分的函数，所以选择交叉熵损失作为损失函数；目标是训练模型使得交叉熵损失最小；

---

## 【6.3】计算分类损失

【test0606_p175_compute_classify_loss_module.py】计算分类损失函数

```python
import torch

# 计算分类损失函数
def compute_classify_loss_batch(input_batch, target_batch, diy_gpt_model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    # 最后一个输出词元的logits
    logits = diy_gpt_model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

# 计算损失加载器
def compute_classify_loss_loader(data_loader, diy_gpt_model, device, num_batches=None):
    total_loss = 0;
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = compute_classify_loss_batch(
                input_batch, target_batch, diy_gpt_model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches
```

---

【test0606_p175_compute_classify_loss_module_main.py】测试案例-计算分类损失函数

```python
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
```

<br>

---

# 【7】在有监督数据上微调模型 

接下来，本文将实现一个训练函数，来微调大模型，目的是最小化训练集损失，提高分类准确率（具体的，提高垃圾消息分类准确率），如图6-15所示；

![image-20250615211209295](./pic/06/0615.png) 

---

<br>

## 【7.1】微调模型以分类垃圾消息

### 【7.1.1】定义训练分类器+定义评估模型函数

【test0607_p177_finetune_gpt_model_module.py】定义训练分类器+定义评估模型函数

```python
import torch

# 微调模型
from src.chapter06.test0606_p175_compute_classify_loss_module import compute_classify_loss_batch, \
    compute_classify_loss_loader
from src.chapter06.test0606_p174_compute_classify_accuracy_module import compute_accuracy_loader

# 定义评估模型函数
def evaluate_model(diy_gpt_model, train_loader, validate_loader, device, eval_iter):
    diy_gpt_model.eval()
    with torch.no_grad():
        train_loss = compute_classify_loss_loader(
            train_loader, diy_gpt_model, device, num_batches=eval_iter
        )
        validate_loss = compute_classify_loss_loader(
            validate_loader, diy_gpt_model, device, num_batches=eval_iter
        )
    diy_gpt_model.train()
    return train_loss, validate_loss


# 定义训练分类器
def train_classifier_simple(
        diy_gpt_model, train_loader, validate_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    # 初始化列表， 以跟踪损失和所见样本
    train_losses, validate_losses, train_accurate_array, validate_accurate_array = [], [], [], []
    examples_seen, global_step = 0, -1

    # 主训练循环
    for epoch in range(num_epochs):
        # 设置模型为训练模式
        diy_gpt_model.train()

        # 遍历每轮的批次
        for input_batch, target_batch in train_loader:
            # 重置上一个批次的损失梯度
            optimizer.zero_grad()
            loss = compute_classify_loss_batch(input_batch, target_batch, diy_gpt_model, device)
            # 计算损失梯度
            loss.backward()
            # 使用损失梯度更新模型权重
            optimizer.step()
            # 跟踪样本，跟踪训练进度
            examples_seen += input_batch.shape[0]
            global_step += 1

            # 可选的评估步骤
            if global_step % eval_freq == 0:
                train_loss, validate_loss = evaluate_model(
                    diy_gpt_model, train_loader, validate_loader, device, eval_iter)
                train_losses.append(train_loss)
                validate_losses.append(validate_loss)
                print(f"Ep {epoch + 1} step {global_step:06d}: "
                      f"train loss = {train_loss:.3f}, "
                      f"validate loss = {validate_loss:.3f}")

        # 每轮训练后计算分类准确率
        train_accuracy = compute_accuracy_loader(
            train_loader, diy_gpt_model, device, num_batches=eval_iter
        )
        validate_accuracy = compute_accuracy_loader(
            validate_loader, diy_gpt_model, device, num_batches=eval_iter
        )
        # 打印分类准确率
        print(f"train_accuracy = {train_accuracy*100:.2f}%")
        print(f"validate_accuracy = {validate_accuracy * 100:.2f}%")

        train_accurate_array.append(train_accuracy)
        validate_accurate_array.append(validate_accuracy)

    return train_losses, validate_losses, train_accurate_array, validate_accurate_array, examples_seen
```

---

### 【7.1.2】绘制分类损失曲线

【test0607_p179_plot_classify_loss_module.py】绘制分类损失曲线

```python
# 绘制分类损失曲线
import matplotlib.pyplot as plt


def plot_values(epoch_seen, examples_seen, train_values, validate_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # 绘制训练集损失和验证集损失与轮数的关系
    ax1.plot(epoch_seen, train_values, label=f"training {label}")
    ax1.plot(epoch_seen, validate_values, linestyle="-.", label=f"validate {label}")

    ax1.set_xlabel("epochs-轮次")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # 为所见样本画第2个x轴
    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlable("examples seen")

    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()
```

---

### 【7.1.3】测试案例-使用分类器及模型评估函数微调模型

```python
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
```

<br>

---

# 【8】使用大模型作为垃圾消息分类器

本文接下来使用微调后的 基于GPT的模型进行垃圾消息分类， 如图6-18。

![image-20250615220514094](./pic/06/0618.png)

---

## 【8.1】使用微调后的模型对文本分类

使用微调后的模型对文本分类的处理步骤： 

- 文本转为词元id； 
- 使用模型预测类别标签；
- 返回相应的类名称；

【test0608_p183_finetuned_model_classify_module.py】使用微调后的模型对文本分类

```python
import torch


# 使用微调后的模型对文本分类
def classify_review(text, diy_gpt_model, tokenizer, device, max_length=None, pad_token_id=50256):
    diy_gpt_model.eval()

    # 准备模型的输入数据
    input_ids = tokenizer.encode(text)
    supprted_context_length = diy_gpt_model.position_emb.weight.shape[0]

    # 截断过长的序列
    input_ids = input_ids[:min(max_length, supprted_context_length)]

    # 填充序列至最长序列长度
    input_ids += [pad_token_id] * (max_length - len(input_ids))

    # 添加批次维度
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    # 推理时，不需要计算梯度
    with torch.no_grad():
        # 最后一个输出词元的logits
        logits = diy_gpt_model(input_tensor)[:, -1, :]

    predicted_label = torch.argmax(logits, dim=-1).item()
    # 返回分类结果
    return "spam" if predicted_label == 1 else "not spam"
```

---

【test0608_p183_finetuned_model_classify_module_main.py】测试案例-使用微调后的模型对文本分类

```python
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
```

<br>

---

# 【9】分类微调大模型小结

微调大模型包含两种策略， 分类微调， 指令微调；

分类微调：通过添加一个小型分类层来替换大模型的输出层； 

在垃圾文本分类模型中，分类层只有2个输出节点（1-垃圾邮件，0-非垃圾邮件）；

与预训练类似，微调的模型输入是将文本转为词元id；

在微调大模型之前，本文把预训练模型加载为基础模型；

分类模型的评估：包括计算分类准确率；

分类模型的微调使用交叉熵损失函数作为损失函数，这与预训练模型相同； 







