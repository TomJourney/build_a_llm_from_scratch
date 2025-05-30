import os
from pathlib import Path
import tiktoken
from src.chapter02.GPTDatasetV1 import GPTDatasetV1 as Diy_GPTDatasetV1
from src.utils import BusiIoUtils
import torch
from torch.utils.data import Dataset, DataLoader

tiktoken_cache_dir = str(Path("..", "tiktoken", ".tiktoken"))
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

print("\n\n=== 使用BPE分词器： tiktoken.get_encoding(\"gpt2\") ")
# 获取BPE分词器
bpeTokenizer = tiktoken.get_encoding("gpt2")


# 创建数据加载器
def create_data_loader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True,
                          num_workers=0):
    # 初始化分词器
    tokenizer = tiktoken.get_encoding("gpt2")
    # 创建数据集
    dataset = Diy_GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # 如果drop_last为True，且批次大小小于指定的batch_size,则会删除最后一批，以防止在训练期间出现损失剧增
        drop_last=drop_last,
        # 用于预处理的cpu进程数
        num_workers=num_workers
    )
    return dataloader


# 读取verdict小说
print("\n\n=== 读取verdict小说，并用自定义分词器BPE分词")
with open(Path(BusiIoUtils.get_root_dir(), "..", "file", "the-verdict.txt")) as f:
    raw_text = f.read()

print(raw_text[:10])
# I HAD alwa

# 创建基于BPE分词器的数据加载器
data_loader = create_data_loader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
# 把dataloader转为python迭代器， 以通过python内置的next()获取下一个条目
data_iter = iter(data_loader)
# 获取下一个条目
# first_batch包含两个张量： 第一个张量存储输入词元id， 第二个张量存储目标词元id
first_batch = next(data_iter)
print("first_batch = ", first_batch)
# first_batch =  [tensor([[  40,  367, 2885, 1464]]), tensor([[ 367, 2885, 1464, 1807]])]
print("\n\n===通过自定义数据加载器(BPE分词器)对应的迭代器获取加载数据集的下一个条目")

# 使用大于1的批次大小(批次大小是超参数，这里设置为8)
data_loader2 = create_data_loader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter2 = iter(data_loader2)
# next(data_iter2)： 加载的数据集的第一个条目包含两个张量： 第一个张量存储输入词元id， 第二个张量存储目标词元id
inputs, targets = next(data_iter2)
print("输入词元id=inputs = \n", inputs)
print("目标词元id=targets = \n", targets)
# 输入词元id=inputs =
#  tensor([[   40,   367,  2885,  1464],
#         [ 1807,  3619,   402,   271],
#         [10899,  2138,   257,  7026],
#         [15632,   438,  2016,   257],
#         [  922,  5891,  1576,   438],
#         [  568,   340,   373,   645],
#         [ 1049,  5975,   284,   502],
#         [  284,  3285,   326,    11]])
# 目标词元id=targets =
#  tensor([[  367,  2885,  1464,  1807],
#         [ 3619,   402,   271, 10899],
#         [ 2138,   257,  7026, 15632],
#         [  438,  2016,   257,   922],
#         [ 5891,  1576,   438,   568],
#         [  340,   373,   645,  1049],
#         [ 5975,   284,   502,   284],
#         [ 3285,   326,    11,   287]])
