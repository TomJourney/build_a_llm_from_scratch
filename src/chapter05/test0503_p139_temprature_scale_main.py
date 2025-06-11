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

