import torch

# 测试案例-为什么要把填充词元50256替换为-100

# 1 计算2个词元的预测logits，logits表示词汇表中每个词元的概率分布的向量
logits_1 = torch.tensor(
    [[-1.0, 1.0],
     [-0.5, 1.5]]
)
# 生成正确的词元索引
targets_1 = torch.tensor([0, 1])
loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
print("计算交叉熵损失, loss_1 = ", loss_1)
# 计算交叉熵损失, loss_1 =  tensor(1.1269)

# 2 计算3个词元的预测logits
logits_2 = torch.tensor(
    [[-1.0, 1.0],
     [-0.5, 1.5],
     [-0.5, 1.5]]
)
# 生成正确的词元索引
targets_2 = torch.tensor([0, 1, 1])
loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
print("计算交叉熵损失, loss_2 = ", loss_2)
# 计算交叉熵损失, loss_2 =  tensor(0.7936)

# 3 把第2步的targets2的第3个目标词元id修改为-100，再计算交叉熵损失
targets_3 = torch.tensor([0, 1, -100])
loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)
print("计算交叉熵损失, loss_3 = ", loss_3)
# 计算交叉熵损失, loss_3 =  tensor(1.1269)， 等于loss_1

# 结论：交叉熵损失计算函数cross_entropy会忽略目标词元id等于-100的交叉熵损失。

