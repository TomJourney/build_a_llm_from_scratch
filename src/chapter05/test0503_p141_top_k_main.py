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

print("top_logits[-1] = ", top_logits[-1])
# top_logits[-1] =  tensor(4.5100)

print("\n 步骤2：把非logits非最高的3个词元的logits值设置为负无穷-inf")
# top_logits[-1] 是最后一个维度
new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],
    input=torch.tensor(float('-inf')).to(next_token_logits.device),
    other=next_token_logits
)
print("设置为负无穷-inf后的logits， new_logits = ", new_logits)
# 设置为负无穷-inf后的logits， new_logits =  tensor([4.5100,   -inf,   -inf, 6.7500,   -inf,   -inf,   -inf, 6.2800,   -inf])

print("\n 步骤3：使用softmax函数把这些值转换为下一个词元的概率(softmax-归一化，值的累加和为1)")
topk_probabilities = torch.softmax(new_logits, dim=0)
print("topk_probabilities = ", topk_probabilities)
# topk_probabilities =  tensor([0.0615, 0.0000, 0.0000, 0.5775, 0.0000, 0.0000, 0.0000, 0.3610, 0.0000])
