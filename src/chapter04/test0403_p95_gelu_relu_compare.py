import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from src.chapter04.test0403_p95_gelu_module import GELU
import torch.nn as nn
import torch

print("\n\n=== 比较GELU与ReLU函数的图像")
gelu, relu = GELU(), nn.ReLU()

# 在-3和3之间创建100个样本数据点
x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)
plt.tight_layout()
plt.show()

