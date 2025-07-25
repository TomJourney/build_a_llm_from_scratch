[TOC]

# 【README】

本文总结自<font color="#ff0000">《从零构建大模型》</font>，非常棒的一本书，墙裂推荐； 

本章中，我们将在无标签数据上预训练大模型，主要内容如下：

- 初始化大模型以进行文本生成；
- 讨论评估生成文本质量的基本方法；
- 计算训练集损失与验证集损失；（评估模型） 
- 大模型训练函数；
- 文本生成策略；
- 权重保存与加载；
- 把OpenAI的预训练权重加载到我们diy的大模型中；

本文代码参见： [https://github.com/TomJourney/build_a_llm_from_scratch](https://github.com/TomJourney/build_a_llm_from_scratch)

---

# 【1】评估文本生成模型

预训练大模型步骤概述，如图5-2所示。

![image-20250609202553999](./pic/05/0502.png) ---

<br>

---

## 【1.1】使用GPT来生成文本

使用DiyGPTModel与GPT_CONFIG_124M来初始化GPT模型。

```python
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

# 实例化gpt模型（使用自定义GPT模型）
torch.manual_seed(123)
gpt_model = DiyGPTModel(GPT_CONFIG_124M)
gpt_model.eval()
```

---

### 【1.1.1】使用GPT模型生成文本过程

使用GPT模型生成文本过程包括3步，如图5-3所示：

- 分词器将输入文本转换为一系列词元id；
- 模型接收这些词元id并生成相应的logits，logits表示词汇表中每个词元的概率分布的向量；
- logits被转换回词元id，分词器把词元id解码为人类可读文本；

![image-20250609203518026](./pic/05/0503.png)

---

【test0501_p119_text_to_token_transfer_util_module.py】用于文本到词元id转换的工具函数 

```python
import torch


# 文本到词元id转换的工具函数
def text_to_tokens_ids(text, tokenizer):
    # 编码，词元化
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    # 使用 .unsqueeze(0) 添加batch维度
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


# 词元id到文本转换的工具函数
def token_ids_to_text(token_ids, tokenizer):
    # 移除batch维度
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

```

【test0501_p119_text_to_token_transfer_util_module_main.py】验证案例-用于文本到词元id转换的工具函数

```python
import torch
import tiktoken
from src.chapter04.test0406_p107_gpt_model_module import DiyGPTModel
from src.chapter04.test0406_p112_gpt_model_generate_text_module import generate_text_simple
from src.chapter05.test0501_p119_text_to_token_transfer_util_module import text_to_tokens_ids, token_ids_to_text

# 文本到词元id转换的工具函数-测试案例
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

# 实例化gpt模型（使用自定义GPT模型）
torch.manual_seed(123)
gpt_model = DiyGPTModel(GPT_CONFIG_124M)
gpt_model.eval()

# 使用文本到词元id转换的工具函数，生成文本
start_context = "Every effort moves you"
# 获取分词器
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    gpt_model=gpt_model,
    index_array=text_to_tokens_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("token_ids = ", token_ids) 
# token_ids =  tensor([[ 6109,  3626,  6100,   345, 34245,  5139]])
print("词元转换为文本工具函数结果 = ", token_ids_to_text(token_ids, tokenizer))
# 词元转换为文本工具函数结果 =  Every effort moves you rentingetic wasnم refres RexMeCHicular stren
```

【代码解说】

显然，上述代码生成的文本是不连贯的，即语句不通。原因是DiyGPTModel没有经过预训练，其权值都是随机的。

接下来，本文会计算生成文本的损失函数大小，这个损失值将作为模型训练的评价指标。

<br>

---

## 【1.2】计算文本生成损失（评估模型）

在GPT模型训练过程中，本文将介绍通过计算文本生成损失值评估文本质量的技术。

文本生成全过程，如图5-4所示，从加载数据到生成文本。

![image-20250609205221880](./pic/05/0504.png)

---

【test0501_p120_text_generate_process_main.py】测试案例-文本生成过程的5个步骤

```python
import tiktoken
import torch

from src.chapter04.test0406_p107_gpt_model_module import DiyGPTModel
from src.chapter05.test0501_p119_text_to_token_transfer_util_module import token_ids_to_text

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

# 实例化gpt模型（使用自定义GPT模型）
torch.manual_seed(123)
diy_gpt_model = DiyGPTModel(GPT_CONFIG_124M)
diy_gpt_model.eval()

# 获取分词器
tokenizer = tiktoken.get_encoding("gpt2")

# 文本生成过程-测试案例
inputs = torch.tensor([[16822, 3626, 6100],  # every effort moves
                       [40, 1107, 588]  # I really like
                       ])
# 与输入匹配，targets是我们期待生成的词元id
targets = torch.tensor([[3626, 6100, 345],  # effort moves you
                        [1107, 588, 11311]  # really like chocolate
                        ])

# 步骤1：为2个输入示例计算logits向量（logits表示词汇表中每个词元的概率分布的向量）
with torch.no_grad():
    logits = diy_gpt_model(inputs)
# 步骤2： 计算每个词元的概率
probability_score = torch.softmax(logits, dim=-1)
print("probability_score.shape = ", probability_score.shape)
# probability_score.shape =  torch.Size([2, 3, 50257])

# 步骤3+步骤4： 使用argmax函数计算概率值最高的索引位置(索引位置就是token的id)
token_ids = torch.argmax(probability_score, dim=-1, keepdim=True)
print("token_ids = ", token_ids)

# 步骤5：把词元id转换回文本
predict_text = token_ids_to_text(token_ids[0].flatten(), tokenizer)
print("预测的文本，predict_text = ", predict_text)
target_text = token_ids_to_text(targets[0], tokenizer)
print("原始的文本， target_text = ", target_text)
# 预测的文本，predict_text =  women saves Admir
# 原始的文本， target_text =   effort moves you
```

【代码解说】

显然，预测结果的文本，与原始的文本存在很大不同，这是模型DiyGPTModel没有训练导致的，即模型使用的权重是随机值，没有经过训练使得损失最小。

损失值（或损失函数值）：表示生成的词元（预测结果）与正确目标（真实值）间的差异，差异越小，则模型效果越高；

<br>

---

### 【1.2.1】模型训练的目标

<font color=red>模型训练的目标：</font>是增加正确目标词元id对应索引位置的softmax概率，即正确位置的概率越高，则模型效果越好，如图5-6所示。

- 或者模型训练的目标是：使得正确的下一次元位置上的概率最大，即最大化正确下一词元的概率；

![image-20250609212557462](./pic/05/0506.png)

【补充】

<font color=red>模型训练的目标是最大化正确下一词元的概率，这涉及到增大其相对于其他词元的概率</font>。通过这种方式，可以确保大模型始终选择正确的词元作为下一词元。

---

### 【1.2.2】计算概率分数的损失

计算概率分数的损失，对概率分数取对数，因为概率分数的对数易于处理，相比于直接处理分数；计算概率分数的损失的步骤，如图5-7所示。

![image-20250609213353940](./pic/05/0507.png)

【图解】

我们的目标是在训练过程中更新模型权重，使得<font color=red>负平均对数概率降到0 </font>。

<font color=red>负平均对数概率计算： 平均对数概率乘以-1 ，该计算方法又称为交叉熵损失 </font>。

- 交叉熵损失：一种常用度量方式，用于衡量两个概率分布的差异； 通过是数据集标签的真实分布与模型生成的预测分布间的差异；
- 交叉熵损失，又称为负平均对数概率；

---

### 【1.2.3】使用torch.cross_entropy函数计算交叉熵损失 

【test0501_p120_text_generate_process_main.py】使用torch.cross_entropy函数计算交叉熵损失

```python
# 打印logits张量与targets张量的形状(logits有3个维度：批处理大小，词元数量，词汇表大小)
# targets张量有2个维度： 批处理大小与词元数量
print("logits.shape = ", logits.shape)
print("targets.shape = ", targets.shape)
# logits.shape =  torch.Size([2, 3, 50257])
# targets.shape =  torch.Size([2, 3])

# 在批处理维度上把logits的维度进行展平（logits的3个维度转为2个维度，targets的2个维度转为1个维度）
logits_flat_predict = logits.flatten(0, 1)
targets_flat = targets.flatten()
print("logits_flat_predict.shape = ", logits_flat_predict.shape)
print("targets_flat.shape = ", targets_flat.shape)
# logits_flat_predict.shape =  torch.Size([6, 50257])
# targets_flat.shape =  torch.Size([6])

print("\n=== 使用torch.cross_entropy函数计算交叉熵损失")
cross_entropy_loss = torch.nn.functional.cross_entropy(logits_flat_predict, targets_flat)
print("交叉熵损失, cross_entropy_loss = ", cross_entropy_loss)
# 交叉熵损失, cross_entropy_loss =  tensor(10.8449)

```

<br>

---

## 【1.3】计算训练集和验证集的损失

具体的，是计算训练集与验证集的交叉熵损失；

【test0501_p127_compute_train_test_loss_module.py】计算训练集加载器与验证集加载器返回的给定批次的交叉熵损失函数+交叉熵损失计算加载器

```python
import torch

# 计算训练集加载器与验证集加载器返回的给定批次的交叉熵损失函数
def compute_loss_batch(input_batch, target_batch, diy_gpt_model, device):
    # to(device) 可以把输入数据转移到GPU上
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = diy_gpt_model(input_batch)
    cross_entropy_loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return cross_entropy_loss

# 交叉熵损失计算加载器
def compute_loss_loader(data_loader, diy_gpt_model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = compute_loss_batch(input_batch, target_batch, diy_gpt_model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches
```

---

【test0501_p127_compute_train_test_loss_module_main.py】计算训练集与测试集间的损失函数-测试案例

```python
from pathlib import Path

import tiktoken
import torch

from src.chapter02.test0206_p35_dataloader import create_data_loader_v1 as create_data_loader_v1
from src.chapter04.test0406_p107_gpt_model_module import DiyGPTModel
from src.utils import BusiIoUtils
from src.chapter05.test0501_p127_compute_train_test_loss_module import compute_loss_batch, compute_loss_loader

# 计算训练集与验证集的交叉熵损失
# 获取分词器
tokenizer = tiktoken.get_encoding("gpt2")

# 读取verdict小说
print("\n\n=== 读取verdict小说")
with open(Path(BusiIoUtils.get_root_dir(), "..", "file", "the-verdict.txt")) as f:
    raw_text = f.read()

total_characters = len(raw_text)
total_token = len(tokenizer.encode(raw_text))
print("总字符个数, total_characters = ", total_characters)
print("总词元个数, total_token = ", total_token)
# 总字符个数, total_characters =  20479
# 总词元个数, total_token =  5145

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
print("\n===train_data_loader = ")
for x, y in train_data_loader:
    print(x.shape, y.shape)
# ===train_data_loader = （训练集有9个批次，每个批次2个样本，每个样本256个词元）
# torch.Size([2, 256]) torch.Size([2, 256])
# torch.Size([2, 256]) torch.Size([2, 256])
# torch.Size([2, 256]) torch.Size([2, 256])
# torch.Size([2, 256]) torch.Size([2, 256])
# torch.Size([2, 256]) torch.Size([2, 256])
# torch.Size([2, 256]) torch.Size([2, 256])
# torch.Size([2, 256]) torch.Size([2, 256])
# torch.Size([2, 256]) torch.Size([2, 256])
# torch.Size([2, 256]) torch.Size([2, 256])

print("\n===test_data_loader = ")
for x, y in test_data_loader:
    print(x.shape, y.shape)
# torch.Size([2, 256]) torch.Size([2, 256]) （测试集有1个批次，每个批次2个样本，每个样本256个词元）

# 实例化gpt模型（使用自定义GPT模型）
torch.manual_seed(123)
diy_gpt_model = DiyGPTModel(GPT_CONFIG_124M)
diy_gpt_model.eval()

# 使用交叉熵损失计算加载器，计算训练集与测试集间的交叉熵损失
print("\n\n=== 使用交叉熵损失计算加载器，计算训练集与测试集间的交叉熵损失")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diy_gpt_model.to(device)
with torch.no_grad():
    train_loss = compute_loss_loader(train_data_loader ,diy_gpt_model, device)
    test_loss = compute_loss_loader(test_data_loader, diy_gpt_model, device)
print("train_loss = ", train_loss)
print("test_loss = ", test_loss)
# train_loss =  10.987583584255642
# test_loss =  10.981106758117676

# 这个损失值就是我们的目标函数，我们训练模型的目的是使得该目标函数值最小
```

---

# 【2】训练大语言模型 

图5-11描述了一个典型的PyTorch神经网络训练工作流程，我们将根据该流程来训练一个大语言模型。

PyTorch神经网络训练工作流程，包括8个步骤。

1. 遍历训练轮次；
2. 在每个训练轮次中遍历批次；
3. 从上一个批次迭代中重置损失梯度；
4. 计算当前批次的损失；
5. 反向传播以计算损失梯度；
6. 使用损失梯度更新模型权重；
7. 打印训练集和验证集的损失；
8. 生成文本样本用于可视化；

![image-20250609223413925](./pic/05/0511.png)

---

## 【2.1】训练大模型代码实现

【test0502_p133_train_model_module.py】训练大模型方法

```python
import torch

from src.chapter04.test0406_p112_gpt_model_generate_text_module import generate_text_simple
from src.chapter05.test0501_p119_text_to_token_transfer_util_module import text_to_tokens_ids, token_ids_to_text
from src.chapter05.test0501_p127_compute_train_test_loss_module import compute_loss_batch, compute_loss_loader


# 打印训练集与测试集的损失
def evaluate_model(diy_gpt_model, train_loader, test_loader, device, eval_iter):
    diy_gpt_model.eval()
    with torch.no_grad():
        train_loss = compute_loss_loader(train_loader, diy_gpt_model, device, num_batches=eval_iter)
        test_loss = compute_loss_loader(test_loader, diy_gpt_model, device, num_batches=eval_iter)
    diy_gpt_model.train()
    return train_loss, test_loss


# 生成文本并打印样本
def generate_and_print_sample(diy_gpt_model, tokenizer, device, start_context):
    diy_gpt_model.eval()
    context_size = diy_gpt_model.position_emb.weight.shape[0]
    encoded = text_to_tokens_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(gpt_model=diy_gpt_model, index_array=encoded, max_new_tokens=50,
                                         context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print("decoded_text = ", decoded_text.replace("\n", " "))
    diy_gpt_model.train()


# 训练模型
def train_model_simple(diy_gpt_model, train_loader, test_loader, optimizer, device, num_epochs, eval_frequency,
                       eval_iter, start_context, tokenizer):
    train_losses, test_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        diy_gpt_model.train()
        for input_batch, target_batch in train_loader:
            # 重置上一个批次迭代中的损失梯度
            optimizer.zero_grad()
            loss = compute_loss_batch(input_batch, target_batch, diy_gpt_model, device)

            # 计算损失梯度
            loss.backward()
            # 使用损失梯度更新模型权重
            optimizer.step()
            global_step += 1

            # 可选的评估步骤
            if global_step % eval_frequency == 0:
                train_loss, test_loss = evaluate_model(diy_gpt_model, train_loader, test_loader, device, eval_iter)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (step {global_step:06d}) :"
                      f"train_loss = {train_loss:.3f}, "
                      f"test_loss = {test_loss:.3f}")
        generate_and_print_sample(diy_gpt_model, tokenizer, device, start_context)
    return train_losses, test_losses, track_tokens_seen

```



---

【test0502_p133_train_model_module_main.py】测试案例-训练大模型

```python
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


```

【代码解说】

根据10轮训练的得到的损失值，本文发现训练集损失显著减少，从到。

测试集损失也迅速减少。但它不会像训练集损失那样变得很小，在第10轮后收敛到6左右。

【结论】

<font color=red>通过观察，验证集损失远远大于训练集损失表明，模型对训练数据过拟合</font>。

<br>

---

# 【3】控制随机性的文本生成策略（解码策略）

本章将介绍<font color=red>用于大模型的文本生成策略，减少训练数据的记忆（避免过拟合）</font>，增加大模型生成文本的独创性； 

具体的，<font color=red>使用两种技术（温度缩放+Top-k采样）改进generate_and_print_sample使用的generate_text_simple()函数</font>。

【test0503_p137_text_generate_strategy_main.py】简要回顾generate_text_simple函数使用大模型生成文本

```python
import tiktoken
import torch

from src.chapter04.test0406_p107_gpt_model_module import DiyGPTModel
from src.chapter04.test0406_p112_gpt_model_generate_text_module import generate_text_simple
from src.chapter05.test0501_p119_text_to_token_transfer_util_module import text_to_tokens_ids, token_ids_to_text

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

# 实例化gpt模型（使用自定义GPT模型）
torch.manual_seed(123)
# 创建自定义gpt模型实例
diy_gpt_model = DiyGPTModel(GPT_CONFIG_124M)

# 获取分词器
tokenizer = tiktoken.get_encoding("gpt2")

# 设置设备为cpu
diy_gpt_model.to("cpu")
# 设置为评估模型，以便关闭如dropout之类的随机组件
diy_gpt_model.eval()

# 使用大模型逐个生成文本
token_ids = generate_text_simple(gpt_model=diy_gpt_model,
                                 index_array=text_to_tokens_ids("Every effort moves you", tokenizer),
                                 max_new_tokens=25,
                                 context_size=GPT_CONFIG_124M["context_length"]
                                 )
print("生成的文本= ", token_ids_to_text(token_ids, tokenizer))
# 生成的文本=  Every effort moves you rentingetic wasnم refres RexMeCHicular stren Mortgage TT remember gard ACTIONSussedOND Land Engeleddedemate breaths proxies GalaxyForm


```

---

## 【3.1】温度缩放

<font color=red>温度缩放定义：一种在下一个词元生成任务中添加概率选择过程的技术</font>。 

- 之前：generate_text_simple()使用torch.argmax(贪婪解码)函数采样具有最高概率的词元作为下一个词元；
- 使用温度缩放技术后：使用一个基于概率分布的采样函数取代argmax；
  - 概率分布：指大模型在每个词元生成步骤为每个词汇表条目生成的概率分数；

---

### 【3.1.1】温度缩放背景

【test0503_p138_temprature_scale_background.py】

```python
import torch

# 温度缩放背景

# 定义一个小型词汇表
small_vocabulary = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8
}
# 对调k-v
inverse_vocabulary = {v: k for k, v in small_vocabulary.items()}
print("inverse_vocabulary = ", inverse_vocabulary)
# inverse_vocabulary =  {0: 'closer', 1: 'every', 2: 'effort', 3: 'forward', 4: 'inches', 5: 'moves', 6: 'pizza', 7: 'toward', 8: 'you'}

# 假设起始上下文为 every effort moves you，并生成下一个词元的 logits ，如下
# logits表示词汇表中每个词元的概率分布的向量
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

print("\n=== 转换为概率分数，并选择概率最大的词汇条目id作为预测的下一个词元id")
probabilities = torch.softmax(next_token_logits, dim=-1)
next_token_id = torch.argmax(probabilities).item()
print("inverse_vocabulary[next_token_id] = ", inverse_vocabulary[next_token_id])
# inverse_vocabulary[next_token_id] =  forward

print("\n=== 使用概率采样，用PyTorch.multinomial替换argmax")
torch.manual_seed(123)
next_token_id_multinomial = torch.multinomial(probabilities, num_samples=1).item()
print("inverse_vocabulary[next_token_id_multinomial] = ", inverse_vocabulary[next_token_id_multinomial])


# inverse_vocabulary[next_token_id_multinomial] =  forward

# torch.multinomial函数按照其概率分数采样下一个词元，重复1000次执行torch.multinomial进行采样，结果如下。
def print_tokens_using_multinomial_sample(probabilities):
    torch.manual_seed(123)
    sample = [torch.multinomial(probabilities, num_samples=1).item() for _ in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, frequency in enumerate(sampled_ids):
        print(f"{frequency} x {inverse_vocabulary[i]}")


print("\n=== 重复1000次执行torch.multinomial进行采样")
print_tokens_using_multinomial_sample(probabilities)
# 73 x closer
# 0 x every
# 0 x effort
# 582 x forward
# 2 x inches
# 0 x moves
# 0 x pizza
# 343 x toward

# 这意味着并不是每次都会选择 forward作为下一个词元，有可能选择 closer 或 inches 或 toward

```

这意味着模型并不是每次都会选择 forward作为下一个词元，有可能选择 closer 或 inches 或 toward；

故本文引入了温度缩放，使用温度缩放可以进一步控制分布和选择过程； 

---

### 【3.1.2】温度缩放

温度缩放：指将logits除以一个大于0的数；

【test0503_p139_temprature_scale_main.py】温度缩放测试案例

```python
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 温度缩放：指将logits除以一个大于0的数

# 定义一个小型词汇表
small_vocabulary = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8
}
# 对调k-v
inverse_vocabulary = {v: k for k, v in small_vocabulary.items()}
print("inverse_vocabulary = ", inverse_vocabulary)
# inverse_vocabulary =  {0: 'closer', 1: 'every', 2: 'effort', 3: 'forward', 4: 'inches', 5: 'moves', 6: 'pizza', 7: 'toward', 8: 'you'}

# 假设起始上下文为 every effort moves you，并生成下一个词元的 logits ，如下
# logits表示词汇表中每个词元的概率分布的向量
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

# 温度缩放函数
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=-0)

# 温度缩放效果画图
temperatures = [1, 0.1, 5]
scaled_probabilities = [softmax_with_temperature(next_token_logits, T)
                        for T in temperatures]
x = torch.arange(len(small_vocabulary))
bar_width = 0.15
fig, ax = plt.subplots(figsize=(5, 3))
for i, T in enumerate(temperatures):
    rects = ax.bar(x + i * bar_width, scaled_probabilities[i], bar_width, label=f'Temperature = {T}')
ax.set_ylabel('Probability')
ax.set_xticks(x)
ax.set_xticklabels(small_vocabulary.keys(), rotation=90)
ax.legend()
plt.tight_layout()
plt.show()
```

【温度缩放效果图，如图5-14所示】 

温度大于1会导致词元概率更加均匀分布。



![image-20250611202110629](./pic/05/0514.png)



<br>

---

## 【3.2】Top-k采样

通过与概率采样和温度缩放相结合，Top-k采样可以改善文本生成结果。

Top-k采样定义：将采样的词元限制在前k个最可能的词元上，并通过掩码概率分数的方式来排除其他词元，如图5-15所示。

![image-20250611203007482](./pic/05/0515.png)

---

【test0503_p141_top_k_main.py】Top-k采样-测试案例

```python
import torch

# top-k采样-测试案例

# 假设起始上下文为 every effort moves you，并生成下一个词元的 logits ，如下
# logits表示词汇表中每个词元的概率分布的向量
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

# 使用 Top-k 采样
print("\n===使用 Top-k(Top-3) 采样")
print("\n 步骤1：选择logits最高的3个词元")
top_k = 3
top_logits, top_positions = torch.topk(next_token_logits, top_k)
print("top_logits = ", top_logits)
print("top_positions = ", top_positions)
# top_logits =  tensor([6.7500, 6.2800, 4.5100])
# top_positions =  tensor([3, 7, 0])

print("\n 步骤2：把非logits非最高的3个词元的logits值设置为负无穷-inf")
# top_logits[-1] 是最后一个元素
new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],
    input=torch.tensor(float('-inf')),
    other=next_token_logits
)
print("设置为负无穷-inf后的logits， new_logits = ", new_logits)
# 设置为负无穷-inf后的logits， new_logits =  tensor([4.5100,   -inf,   -inf, 6.7500,   -inf,   -inf,   -inf, 6.2800,   -inf])

print("\n 步骤3：使用softmax函数把这些值转换为下一个词元的概率(softmax-归一化，值的累加和为1)")
topk_probabilities = torch.softmax(new_logits, dim=0)
print("topk_probabilities = ", topk_probabilities)
# topk_probabilities =  tensor([0.0615, 0.0000, 0.0000, 0.5775, 0.0000, 0.0000, 0.0000, 0.3610, 0.0000])

```

---

## 【5.3】基于温度缩放与Top-k采样修改文本生成函数

【test0503_p142_modify_text_generate_function.py】基于温度缩放与Top-k采样修改文本生成函数-generate_text_simple()

```python
import torch


# 基于温度缩放与Top-k采样修改文本生成函数-generate_text_simple()
# 生成文本（index_array是当前文本的索引数组，形状为(batch, n_tokens)）
def based_temperature_topk_generate_text_simple(gpt_model, index_array, max_new_tokens, context_size, temperature=0.0,
                                                top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        # 把当前文本截断至支持的长度。若大模型仅支持5个词元，但输入文本长度为10，则只有最后5个词元被用作输入文本
        sub_input_index_array = index_array[:, -context_size:]
        with torch.no_grad():
            logits = gpt_model(sub_input_index_array)

        # 只关注最后一个输出的内容，因为形状会从 (batch, n_token, vocab_size) 变为 (batch, vocab_size)
        logits = logits[:, -1, :]

        # 使用top-k采样筛选logits
        if top_k is not None:
            top_logit, _ = torch.topk(logits, top_k)
            min_value = top_logit[:, -1]
            logits = torch.where(
                condition=logits < min_value,
                input=torch.tensor(float('-inf')),
                other=logits
            )
        # 使用温度缩放
        if temperature > 0.0:
            logits = logits / temperature
            probabilities = torch.softmax(logits, dim=-1)
            index_next = torch.multinomial(probabilities, num_samples=1)
        # 当禁用温度缩放时，则执行贪心解码，选取下一个词元
        else:
            index_next = torch.argmax(logits, dim=-1, keepdim=True)
        if index_next == eos_id:
            break
        index_array = torch.cat((index_array, index_next), dim=1)
    return index_array

```

---

### 【5.3.1】基于温度缩放与Top-k采样修改文本生成函数-测试案例

【test0503_p142_modify_text_generate_function_main.py】基于温度缩放与Top-k采样修改文本生成函数-测试案例

```python
import tiktoken
import torch

from src.chapter04.test0406_p107_gpt_model_module import DiyGPTModel
from src.chapter05.test0501_p119_text_to_token_transfer_util_module import text_to_tokens_ids, token_ids_to_text
from src.chapter05.test0503_p142_modify_text_generate_function import based_temperature_topk_generate_text_simple

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

# 实例化gpt模型（使用自定义GPT模型）
torch.manual_seed(123)
# 创建自定义gpt模型实例
diy_gpt_model = DiyGPTModel(GPT_CONFIG_124M)

# 获取分词器
tokenizer = tiktoken.get_encoding("gpt2")

# 设置设备为cpu
diy_gpt_model.to("cpu")
# 设置为评估模型，以便关闭如dropout之类的随机组件
diy_gpt_model.eval()

print("\n\n=== 应用基于温度缩放与Top-k采样的文本生成函数生成文本")
torch.manual_seed(123)
token_ids = based_temperature_topk_generate_text_simple(
    gpt_model=diy_gpt_model,
    index_array=text_to_tokens_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)
text_generate_result = token_ids_to_text(token_ids, tokenizer)
print("生成结果， text_generate_result = ", text_generate_result)
# text_generate_result =  Every effort moves youEveryiliaralso stabbed OrleansAllowsean 52anche crime winter unbeaten quoteembedreportprint earning
```

<br>

---

# 【4】使用PyTorch加载和保存模型权重

使用PyTorch加载和保存模型权重，目的是无需二次训练模型参数；

- 保存权重：torch.save()
- 加载权重：torch.load()
- 恢复模型权重：diy_gpt_model2.load_state_dict()
- 恢复优化器权重：loaded_optimizer.load_state_dict()

---

## 【4.1】代码实现

【test0504_p144_save_model_weight.py】使用PyTorch加载和保存模型权重

```python
import torch

from src.chapter04.test0406_p107_gpt_model_module import DiyGPTModel

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

# 实例化gpt模型（使用自定义GPT模型）
torch.manual_seed(123)
# 创建自定义gpt模型实例
diy_gpt_model = DiyGPTModel(GPT_CONFIG_124M)
# 获取设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n\n=== step1：保存模型权重，推荐使用torch.save保存模型的state_dict")
torch.save(diy_gpt_model.state_dict(), "diy_gpt_model_weights.pth")
# diy_gpt_model.pth是保存state_dict的文件名，其中 pth是PyTorch文件的扩展名

print("\n\n=== step2：加载模型权重，使用torch.load()")
loaded_diy_gpt_model = DiyGPTModel(GPT_CONFIG_124M)
loaded_diy_gpt_model.load_state_dict(torch.load("diy_gpt_model_weights.pth", map_location=device))
# loaded_diy_gpt_model.eval() 把模型切换为推断模式 ，这会禁用模型的dropout层
loaded_diy_gpt_model.eval()

# 像 AdamW这样的自适应优化器可以为每个模型权重存储额外的参数
# AdamW可以使用历史数据动态调整每个模型参数的学习率，如果没有它，优化器就会被重置，模型学习效果可能不佳

# 使用AdamW优化器和train_model_simple函数，对模型实例diy_gpt_model进行10轮训练
# .parameters()方法返回模型的所有可训练权重参数
optimizer = torch.optim.AdamW(
    diy_gpt_model.parameters(), lr=0.0004, weight_decay=0.1)
print("\n\n=== step3：使用torch.save()保存模型和优化器的state_dict内容")
torch.save({
        "model_state_dict": diy_gpt_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    },
    "model_and_optimizer_weights.pth"
)


print("\n\n=== step4：使用torch.load()加载保存的数据，再使用 load_state_dict恢复模型和优化器的状态")
checkpoint = torch.load("model_and_optimizer_weights.pth", map_location=device)
diy_gpt_model2 = DiyGPTModel(GPT_CONFIG_124M)
# 恢复模型权重
diy_gpt_model2.load_state_dict(checkpoint["model_state_dict"])
# 创建优化器
loaded_optimizer = torch.optim.AdamW(diy_gpt_model2.parameters(), lr=5e-4, weight_decay=0.1)
# 恢复优化器权重
loaded_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
diy_gpt_model2.train()

```

<br>

---

# 【5】从OpenAI加载预训练权重 

OpenAI公开分享了GPT-2模型的权重，我们无需重新训练模型。

我们可以把OpenAI的模型权重加载到DiyGPTModel中，并使用该模型进行文本生成。

- 模型权重：指存储在PyTorch的Linear层和Embedding层的.weight属性中的权重参数；（前面在训练模型时，我们通过model.parameters()访问过权重参数）

## 【5.1】下载OpenAI通过TensorFlow保存的GPT-2模型权重

【安装工具】

```
pip install tensorflow>=2.15.0 tqdm>=4.66 # 2.15.0安装报错，设置tensorflow==2.16.2
```

【下载加载gpt-2架构设置与权重参数的python模块（本书作者提供）】

```python
# 下载加载gpt-2架构设置与权重参数的python模块（本书作者提供）
# print("\n step1：下载gpt-download.py")
# url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch05/01_main-chapter-code/gpt_download.py"
# filename = url.split("/")[-1]
# urllib.request.urlretrieve(url, filename)
```

【下载的gpt_download.py】 

```python
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch


import os
import urllib.request

# import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def download_and_load_gpt2(model_size, models_dir):
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination, backup_url=None):
    def _attempt_download(download_url):
        with urllib.request.urlopen(download_url) as response:
            # Get the total file size from headers, defaulting to 0 if not present
            file_size = int(response.headers.get("Content-Length", 0))

            # Check if file exists and has the same size
            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"File already exists and is up-to-date: {destination}")
                    return True  # Indicate success without re-downloading

            block_size = 1024  # 1 Kilobyte

            # Initialize the progress bar with total file size
            progress_bar_description = os.path.basename(download_url)
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
                with open(destination, "wb") as file:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        file.write(chunk)
                        progress_bar.update(len(chunk))
            return True

    try:
        if _attempt_download(url):
            return
    except (urllib.error.HTTPError, urllib.error.URLError):
        if backup_url is not None:
            print(f"Primary URL ({url}) failed. Attempting backup URL: {backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except urllib.error.HTTPError:
                pass

        # If we reach here, both attempts have failed
        error_message = (
            f"Failed to download from both primary URL ({url})"
            f"{' and backup URL (' + backup_url + ')' if backup_url else ''}."
            "\nCheck your internet connection or the file availability.\n"
            "For help, visit: https://github.com/rasbt/LLMs-from-scratch/discussions/273"
        )
        print(error_message)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Alternative way using `requests`
"""
def download_file(url, destination):
    # Send a GET request to download the file in streaming mode
    response = requests.get(url, stream=True)

    # Get the total file size from headers, defaulting to 0 if not present
    file_size = int(response.headers.get("content-length", 0))

    # Check if file exists and has the same size
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"File already exists and is up-to-date: {destination}")
            return

    # Define the block size for reading the file
    block_size = 1024  # 1 Kilobyte

    # Initialize the progress bar with total file size
    progress_bar_description = url.split("/")[-1]  # Extract filename from URL
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # Open the destination file in binary write mode
        with open(destination, "wb") as file:
            # Iterate over the file data in chunks
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))  # Update progress bar
                file.write(chunk)  # Write the chunk to the file
"""


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params

```



【test0505_p146_download_and_import_openai_gpt2.py】从gpt-download.py中 导入 download_and_load_gpt2函数，加载gpt-2架构设置与权重参数到python会话中

```python
import urllib.request
from gpt_download import download_and_load_gpt2

# 下载加载gpt-2架构设置与权重参数的python模块（本书作者提供）
# print("\n step1：下载gpt-download.py")
# url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch05/01_main-chapter-code/gpt_download.py"
# filename = url.split("/")[-1]
# urllib.request.urlretrieve(url, filename)

# 从gpt-download.py中 导入 download_and_load_gpt2函数
print("\n step2：使用 download_and_load_gpt2函数 加载gpt-2架构设置和权重参数到python会话中")
settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
)
# 执行上述代码-download_and_load_gpt2 函数， 将下载参数量为1.24亿的GPT-2模型的7个文件
print("\n=== 执行上述代码-download_and_load_gpt2函数， 将下载参数量为1.24亿的GPT-2模型的7个文件，， 下载完成")

# 打印download_and_load_gpt2函数 加载gpt-2架构设置和权重参数
print("settings = ", settings)
print("params(部分代码) = ", params) 
# settings =  {'n_vocab': 50257, 'n_ctx': 1024, 'n_embd': 768, 'n_head': 12, 'n_layer': 12}
# params =  {'blocks': [{'attn': {'c_attn': {'b': array([ 0.48033914, -0.5254326 , -0.42926455, ...,  0.01257301,
#        -0.04987717,  0.00324764], dtype=float32), 'w': array([[-0.4738484 , -0.26136586, -0.09780374, ...,  0.05132535,
#         -0.0584389 ,  0.02499568] .....
```

【下载文件示例】

![image-20250614092123993](./pic/05/051502-p146.png)

<br>

----

### 【5.1.1】GPT-2模型的组成

【settings字典】：GPT-2大模型架构的设置，如GPT_CONFIG_124M。

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

而GPT2的settings如下：

```python
{'n_vocab': 50257, 'n_ctx': 1024, 'n_embd': 768, 'n_head': 12, 'n_layer': 12}
```

<br>

【params字典】GPT-2大模型的权重参数，包含实际的权重张量。

````python
print("gpt2_params.keys() = ", gpt2_params.keys())
print("gpt2_params[\"wte\"]", gpt2_params["wte"])
# gpt2_params.keys() =  dict_keys(['blocks', 'b', 'g', 'wpe', 'wte'])
# gpt2_params["wte"] [[-0.11010301 -0.03926672  0.03310751 ... -0.1363697   0.01506208
#    0.04531523]
#  [ 0.04034033 -0.04861503  0.04624869 ...  0.08605453  0.00253983
#    0.04318958]
#  [-0.12746179  0.04793796  0.18410145 ...  0.08991534 -0.12972379
#   -0.08785918]
#  ...
#  [-0.04453601 -0.05483596  0.01225674 ...  0.10435229  0.09783269
#   -0.06952604]
#  [ 0.1860082   0.01665728  0.04611587 ... -0.09625227  0.07847701
#   -0.02245961]
#  [ 0.05135201 -0.02768905  0.0499369  ...  0.00704835  0.15519823
#    0.12067825]]
````

【代码解说】wte：表示词元嵌入层的权重；

<br>

---

## 【5.2】应用GPT-2模型参数到DiyGPTModel

【test0505_p148_load_gpt2_params_to_diy_gpt_model_module.py】把gpt2参数加载到自定义gpt模型的模块

```python
import torch.nn
import numpy as np


# 把right张量返回为可训练的PyTorch参数
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch left.shape = {left.shape}, right.shape = {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

# 把params字典中的权重加载到 DiyGPTModel 实例中
# 把模型的位置信息和词元嵌入权重设置为params中指定的值
def load_weights_into_gpt(diy_gpt_model, params):
    diy_gpt_model.position_emb.weights = assign(diy_gpt_model.position_emb.weight, params['wpe'])
    diy_gpt_model.token_emb.weights = assign(diy_gpt_model.token_emb.weight, params['wte'])

    for b_index in range(len(params["blocks"])):
        # 1 使用gpt模型参数params中注意力的注意力权重替换diy_gpt_model模型对应参数
        q_w, k_w, v_w = np.split(
            (params["blocks"][b_index]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        diy_gpt_model.transformer_blocks[b_index].attention.W_query.weight = assign(
            diy_gpt_model.transformer_blocks[b_index].attention.W_query.weight, q_w.T
        )
        diy_gpt_model.transformer_blocks[b_index].attention.W_key.weight = assign(
            diy_gpt_model.transformer_blocks[b_index].attention.W_key.weight, k_w.T
        )
        diy_gpt_model.transformer_blocks[b_index].attention.W_value.weight = assign(
            diy_gpt_model.transformer_blocks[b_index].attention.W_value.weight, v_w.T
        )

        # 2 使用gpt模型参数params中注意力的偏置权重替换diy_gpt_model模型对应参数
        q_b, k_b, v_b = np.split(
            (params["blocks"][b_index]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        diy_gpt_model.transformer_blocks[b_index].attention.W_query.bias = assign(
            diy_gpt_model.transformer_blocks[b_index].attention.W_query.bias, q_b
        )
        diy_gpt_model.transformer_blocks[b_index].attention.W_key.bias = assign(
            diy_gpt_model.transformer_blocks[b_index].attention.W_key.bias, k_b
        )
        diy_gpt_model.transformer_blocks[b_index].attention.W_value.bias = assign(
            diy_gpt_model.transformer_blocks[b_index].attention.W_value.bias, v_b
        )

        # 3 使用gpt模型参数params中注意力的投影层权重替换diy_gpt_model模型对应参数
        diy_gpt_model.transformer_blocks[b_index].attention.out_proj.weight = assign(
            diy_gpt_model.transformer_blocks[b_index].attention.out_proj.weight,
            params["blocks"][b_index]["attn"]["c_proj"]["w"].T
        )
        diy_gpt_model.transformer_blocks[b_index].attention.out_proj.bias = assign(
            diy_gpt_model.transformer_blocks[b_index].attention.out_proj.bias,
            params["blocks"][b_index]["attn"]["c_proj"]["b"]
        )
        diy_gpt_model.transformer_blocks[b_index].feed_forward.layers[0].weight = assign(
            diy_gpt_model.transformer_blocks[b_index].feed_forward.layers[0].weight,
            params["blocks"][b_index]["mlp"]["c_fc"]["w"].T
        )

        # 4 使用gpt模型参数params中权重替换diy_gpt_model模型对应参数
        diy_gpt_model.transformer_blocks[b_index].feed_forward.layers[0].bias = assign(
            diy_gpt_model.transformer_blocks[b_index].feed_forward.layers[0].bias,
            params["blocks"][b_index]["mlp"]["c_fc"]["b"]
        )
        diy_gpt_model.transformer_blocks[b_index].feed_forward.layers[2].weight = assign(
            diy_gpt_model.transformer_blocks[b_index].feed_forward.layers[2].weight,
            params["blocks"][b_index]["mlp"]["c_proj"]["w"].T
        )
        diy_gpt_model.transformer_blocks[b_index].feed_forward.layers[2].bias = assign(
            diy_gpt_model.transformer_blocks[b_index].feed_forward.layers[2].bias,
            params["blocks"][b_index]["mlp"]["c_proj"]["b"]
        )

        # 5 使用gpt模型参数params中归一化参数替换diy_gpt_model模型对应参数
        diy_gpt_model.transformer_blocks[b_index].layer_norm1.scale = assign(
            diy_gpt_model.transformer_blocks[b_index].layer_norm1.scale,
            params["blocks"][b_index]["ln_1"]["g"]
        )
        diy_gpt_model.transformer_blocks[b_index].layer_norm1.shift = assign(
            diy_gpt_model.transformer_blocks[b_index].layer_norm1.shift,
            params["blocks"][b_index]["ln_1"]["b"]
        )
        diy_gpt_model.transformer_blocks[b_index].layer_norm2.scale = assign(
            diy_gpt_model.transformer_blocks[b_index].layer_norm2.scale,
            params["blocks"][b_index]["ln_2"]["g"]
        )
        diy_gpt_model.transformer_blocks[b_index].layer_norm2.shift = assign(
            diy_gpt_model.transformer_blocks[b_index].layer_norm2.shift,
            params["blocks"][b_index]["ln_2"]["b"]
        )
        diy_gpt_model.final_norm.scale = assign(diy_gpt_model.final_norm.scale, params["g"])
        diy_gpt_model.final_norm.shift = assign(diy_gpt_model.final_norm.shift, params["b"])
        diy_gpt_model.out_head.weight = assign(diy_gpt_model.out_head.weight, params["wte"])
```

【test0505_p148_load_gpt2_params_to_diy_gpt_model_module_main.py】 测试案例-把gpt2参数加载到自定义gpt模型的模块

```python
import tiktoken
import torch.nn

from gpt_download import download_and_load_gpt2
from src.chapter04.test0406_p107_gpt_model_module import DiyGPTModel
from src.chapter05.test0501_p119_text_to_token_transfer_util_module import text_to_tokens_ids, token_ids_to_text
from src.chapter05.test0503_p142_modify_text_generate_function import based_temperature_topk_generate_text_simple
from src.chapter05.test0505_p148_load_gpt2_params_to_diy_gpt_model_module import load_weights_into_gpt

# 应用GPT-2模型参数到DiyGPTModel

# 通过download_and_load_gpt2函数导入
print("\n step2：使用 download_and_load_gpt2函数 加载gpt-2架构设置和权重参数到python会话中")
gpt2_settings, gpt2_params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2", is_download=False
)

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

# 不同大小的GPT2大模型的参数量从1.24亿到15.58亿不等。
# 核心架构相同，唯一区别是嵌入层大小-emb_dim， 以及诸如注意力头-n_heads， Transformer块的重复次数-n_layers不同
# 字典保存不同模型尺寸的GPT模型参数
gpt2_model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (744M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25}
}
# 选择gpt2-small模型参数
model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(gpt2_model_configs[model_name])
# OpenAI的原始GPT-2模型使用1024个词元长度进行训练
NEW_CONFIG.update({"context_length": 1024})
# OpenAI在多头注意力模块的线性层中使用偏置向量实现查询矩阵，键矩阵和值矩阵的计算
NEW_CONFIG.update({"qkv_bias": True})

# 创建diy大模型实例
diy_gpt_model_based_gpt2_config = DiyGPTModel(NEW_CONFIG)
diy_gpt_model_based_gpt2_config.eval()

# GPT模型实例使用随机权重初始化，所以最后一步，我们需要用gpt2模型的权重覆盖随机权重
load_weights_into_gpt(diy_gpt_model_based_gpt2_config, gpt2_params)
# 获取设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diy_gpt_model_based_gpt2_config.to(device)

# 获取分词器
tokenizer = tiktoken.get_encoding("gpt2")

# 传入使用gpt2模型参数的大模型到文本生成函数， 生成新文本
torch.manual_seed(123)
token_ids = based_temperature_topk_generate_text_simple(
    gpt_model=diy_gpt_model_based_gpt2_config,
    index_array=text_to_tokens_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
)
print("生成的新文本 = ", token_ids_to_text(token_ids, tokenizer))

```

【代码解说】

上述模型能够生成连贯的文本。

不过在这个生成文本的过程中，一个小错误也可能导致模型生成文本失败或不连贯。

第6章，本文将进一步使用这个预训练模型， 并对其进行微调，以分类文本和遵循指令； 

<br>

---

# 【6】总结

当大模型生成文本时，是逐个生成下一个词元而迭代进行的；

默认情况下，下一个词元是通过将模型输出转换为概率分数，并从词汇表中选择与最高概率分数对应的词元来生成的，这称为<font color=red>贪婪解码</font>；

通过使用概率采样和温度缩放，可以干预生成文本的多样性与连贯性；

在训练过程中， 训练集损失与验证集损失，可用于衡量大模型生成的文本质量；

对大模型进行预训练，涉及改变其参数权重以最小化训练损失； 

大模型的训练循环是深度学习中的一个标准过程，使用了传统的交叉熵损失和AdamW优化器； 

在大型文本语料库上预训练大模型既耗时又耗资源，可以加载公开可用的权重作为大数据集上自行进行预训练的替代方案；

<br>





