import torch

input_ids = torch.tensor([2, 3, 5, 1])

vocab_size = 6
output_doim = 3

# 实例化一个嵌入层
torch.manual_seed(123)
embeding_layer = torch.nn.Embedding(vocab_size, output_doim)

# 打印权重矩阵， 其中每一行对应词汇表中的一个词元，每一列对应一个嵌入维度
print("\n\n===打印权重矩阵 6*3 ")
print(embeding_layer.weight)
# tensor([[ 0.3374, -0.1778, -0.1690],
#         [ 0.9178,  1.5810,  1.3010],
#         [ 1.2753, -0.2010, -0.1606],
#         [-0.4015,  0.9666, -1.1481],
#         [-1.1589,  0.3255, -0.6315],
#         [-2.8400, -0.7849, -1.4096]], requires_grad=True)

# 获得某一个词元id=3的嵌入向量
print("\n\n===获得某一个词元的嵌入向量")
print(embeding_layer(torch.tensor([3])))
# tensor([[-0.4015,  0.9666, -1.1481]], grad_fn=<EmbeddingBackward0>)