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

