import torch
from pathlib import Path
from src.utils import BusiIoUtils
from src.chapter02.test0206_p35_dataloader import create_data_loader_v1 as create_data_loader_v1

# 创建嵌入层
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
# 读取verdict小说
print("\n\n=== 读取verdict小说，并用自定义分词器分词")
with open(Path(BusiIoUtils.get_root_dir(), "..", "file", "the-verdict.txt")) as f:
    raw_text = f.read()

print("\n\n=== create_data_loader_v1() 创建数据加载器")
data_loader = create_data_loader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(data_loader)
# input是第一批词元id数组，每批大小为8行
inputs, targets = next(data_iter)
print("\n inputs=", inputs)
print("\n inputs.shape=", inputs.shape)

# inputs = tensor([[40, 367, 2885, 1464],
#                  [1807, 3619, 402, 271],
#                  [10899, 2138, 257, 7026],
#                  [15632, 438, 2016, 257],
#                  [922, 5891, 1576, 438],
#                  [568, 340, 373, 645],
#                  [1049, 5975, 284, 502],
#                  [284, 3285, 326, 11]])
#
# inputs.shape = torch.Size([8, 4])

# 使用嵌入层把这些词元id嵌入256维的向量中
print("\n===使用嵌入层把这些词元id嵌入256维的向量中， 批次大小=8，每个批次有4个词元，故得到8*4*256的张量（多维数组）")
token_embeddings = token_embedding_layer(inputs)
print("\ntoken_embeddings.shape = ", token_embeddings.shape)
# token_embeddings.shape =  torch.Size([8, 4, 256])

# 创建一个维度与 token_embedding_layer 相同的嵌入层
print("\n=== 创创建一个维度与 token_embedding_layer 相同的位置嵌入层-position_embeddings_layer")
context_length = max_length # 4
position_embeddings_layer = torch.nn.Embedding(context_length, output_dim)
position_embeddings = position_embeddings_layer(torch.arange(context_length))
print("\nposition_embeddings.shape = ", position_embeddings.shape)
# position_embeddings.shape =  torch.Size([4, 256])

# pytorch会在每个批次中的每个 4*256 维的词元嵌入张量上都添加一个 4*256 维度的pos_embeddings张量：
print("\n=== pytorch会在每个批次中的每个 4*256 维的词元嵌入张量上都添加一个 4*256 维度的pos_embeddings张量：")
input_embeddings = token_embeddings + position_embeddings
print("\ninput_embeddings = token_embeddings + position_embeddings")
print("\ninput_embeddings.shape = ", input_embeddings.shape)
# input_embeddings.shape =  torch.Size([8, 4, 256])
print("\n备注：1.张量就是多维数组（向量或向量的向量）； 2.嵌入或embedding就是向量")