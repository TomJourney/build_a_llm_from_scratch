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

## 【5.1】使用GPT来生成文本

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

### 【5.1.1】使用GPT模型生成文本过程

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
print("词元转换为文本工具函数结果 = ", token_ids_to_text(token_ids, tokenizer))
# 词元转换为文本工具函数结果 =  Every effort moves you rentingetic wasnم refres RexMeCHicular stren
```



