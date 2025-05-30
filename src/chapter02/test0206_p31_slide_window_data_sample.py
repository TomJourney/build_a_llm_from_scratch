import os
from pathlib import Path

import tiktoken

from src.utils import BusiIoUtils

tiktoken_cache_dir = str(Path("..", "tiktoken", ".tiktoken"))
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

print("\n\n=== 使用BPE分词器 ")
# 获取BPE分词器
bpeTokenizer = tiktoken.get_encoding("gpt2")

# 读取verdict小说
print("\n\n=== 读取verdict小说，并用自定义分词器分词")
with open(Path(BusiIoUtils.get_root_dir(), "..", "file", "the-verdict.txt")) as f:
    raw_text = f.read()
encoded_text = bpeTokenizer.encode(raw_text)
print(len(encoded_text))
# 5145

# 截取前50个词元
encode_sample = encoded_text[50:]

# 创建下一个单词预测任务的输入-目标对
context_size = 4  # 滑动窗口大小=4
x = encode_sample[:context_size]
y = encode_sample[1:context_size + 1]
print(f"x:{x}")
print(f"y:      {y}")
# x:[290, 4920, 2241, 287]
# y:      [4920, 2241, 287, 257]

# 处理多个预测任务
print("\n\n===处理多个预测任务，输出词元id")
for i in range(1, context_size + 1):
    context = encode_sample[:i]
    desired = encode_sample[i]
    print(context, "---->", desired)
# [290] ----> 4920
# [290, 4920] ----> 2241
# [290, 4920, 2241] ----> 287
# [290, 4920, 2241, 287] ----> 257

# 处理多个预测任务
print("\n\n===处理多个预测任务，输出词元id解码后的词元")
for i in range(1, context_size + 1):
    context = encode_sample[:i]
    desired = encode_sample[i]
    print(bpeTokenizer.decode(context), "---->", bpeTokenizer.decode([desired]))
# and ---->  established
# and established ---->  himself
# and established himself ---->  in
# and established himself in ---->  a

# 实现一个数据加载器
# 目标是： 返回两个张量， 一个包含大模型所见的文本输入的输入张量，另一个包含大模型需要预测的目标词元的目标张量
import torch






