[TOC]

# 【README】

本文总结自<font color="#ff0000">《从零构建大模型》</font>，非常棒的一本书，墙裂推荐； 

本文实现微调大模型以遵循人类指令，如图7-1所示。

<font color=red>图7-1显式了微调大模型主要有两种方式</font>：

- 用于文本分类的微调（分类微调）；
- 遵循人类指令的微调（指令微调）； 

![image-20250618193121781](./pic/07/0701.png)

---

# 【1】指令微调介绍

指令微调：提高大模型遵循指令并生成合理回复的能力。（准备数据集是指令微调的关键部分）

指令微调的步骤（如图7-3）：

1. 第1阶段：准备数据集；
   1. 下载和制作数据集；
   2. 数据集分批；
   3. 创建数据加载器；
2. 第2阶段：微调大模型；
   1. 加载预训练的大模型；
   2. 指令微调大模型；
   3. 检查模型损失；
3. 第3阶段：评估大模型；
   1. 提取回复；
   2. 量化评估；
   3. 对回复打分；

![image-20250618194125206](./pic/07/0703.png)

<br>

---

# 【2】为指令微调准备数据集

【下载数据集】

下载地址：[https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json](https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json)

【test0702_p187_download_dataset_main.py】测试案例-下载数据集

```python
import json

# 下载并打印数据集
# url=https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json

with open('./dataset/instruction-data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

print("数据量 = ", len(data))
# 数据量 =  1100
# 打印某条数据
print("data[50] = ", data[50])
# data[50] =  {'instruction': 'Identify the correct spelling of the following word.', 'input': 'Ocassion', 'output': "The correct spelling is 'Occasion.'"}

# {
# 	"instruction": "Identify the correct spelling of the following word.",
# 	"input": "Ocassion",
# 	"output": "The correct spelling is Occasion."
# }

# 打印数据项=data[999]
print("data[999] = ", data[999])
# data[999] =  {'instruction': "What is an antonym of 'complicated'?", 'input': '', 'output': "An antonym of 'complicated' is 'simple'."}

# {
# 	"instruction": "What is an antonym of 'complicated'?",
# 	"input": "",
# 	"output": "An antonym of "complicated" is 'simple'."
# }
```

【代码解说】并不是所有数据样本的input属性都有值； 

<br>

---

## 【2.1】指令微调的数据样本格式

指令微调需要在一个提供输入-输出对（如data[500]中的input，output）的数据集上训练模型。 

大模型训练所需的样本格式如图7-4所示，也被称为提示词风格；

考虑Alpaca提示词风格很大程度上奠定了指令微调的基础，故<font color=red>本文后续部分均使用Alpaca提示词风格</font>。

![image-20250618201217766](D:\studynote\00-ai-llm\00-01-build_a_large_language_model\study-note\pic\07\0704.png)

---

## 【2.2】原生数据样本转为Alpaca提示词风格的样本

【test0702_p189_format_input_to_alpaca_module.py】Alpaca提示词风格转换函数

```python
def format_input_to_alpaca(entry):
    instruction_text = (
        f"Below is an instruction that describes a task."
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )

    return instruction_text + input_text
```

【test0702_p189_format_input_to_alpaca_module_main.py】测试案例-Alpaca提示词风格转换函数

```python
import json

from src.chapter07.test0702_p189_format_input_to_alpaca_module import format_input_to_alpaca

# 读取数据集
with open('./dataset/instruction-data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

print("\n=== 调用 format_input_to_alpaca 把数据data[50]转为 Alpaca提示词格式")
model_input = format_input_to_alpaca(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"
print(model_input + desired_response)

# Below is an instruction that describes a task.Write a response that appropriately completes the request.
#
# ### Instruction:
# Identify the correct spelling of the following word.
#
# ### Input:
# Ocassion
#
# ### Response:
# The correct spelling is 'Occasion.'

print("\n=== 调用 format_input_to_alpaca 把数据data[999]转为 Alpaca提示词格式")
model_input = format_input_to_alpaca(data[999])
desired_response = f"\n\n### Response:\n{data[999]['output']}"
print(model_input + desired_response)

# Below is an instruction that describes a task.Write a response that appropriately completes the request.
#
# ### Instruction:
# What is an antonym of 'complicated'?
#
# ### Response:
# An antonym of 'complicated' is 'simple'.
```

【代码解说】

data[999]没有input，转为Alpaca提示词风格后也没有对应的input。

---

## 【2.3】划分数据集

把数据集划分为训练集、验证集、测试集。

【test0702_p190_split_dataset_main.py】测试案例-划分数据集

```python
import json

from src.chapter07.test0702_p189_format_input_to_alpaca_module import format_input_to_alpaca

# 读取数据集
with open('./dataset/instruction-data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

print("\n=== 划分数据集： 训练集85%， 测试集10%， 验证集5%")
train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
validate_portion = int(len(data) * 0.05)

train_data = data[:train_portion]
test_data = data[train_portion:train_portion+test_portion]
validate_data = data[train_portion+test_portion:]

print("训练集长度 = ", len(train_data))
print("测试集长度 = ", len(test_data))
print("验证集长度 = ", len(validate_data))
# 训练集长度 =  935
# 测试集长度 =  110
# 验证集长度 =  55
```

<br>

---

# 【3】将数据组织成训练批次（数据集分批）







