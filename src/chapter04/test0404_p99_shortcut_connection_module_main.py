import torch
import torch.nn as nn
from src.chapter04.test0404_p99_shortcut_connection_module import ExampleDeepNeuralNetwork

layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)

# 定义在模型的反向传播过程中计算梯度的函数
def print_gradient(model, x):
    # 前向传播
    output = model(x)
    target = torch.tensor([[0.]])

    # 计算损失
    loss = nn.MSELoss()
    loss = loss(output, target)

    # 使用反向传播来计算梯度
    loss.backward()

    # 打印梯度
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient name of {param.grad.abs().mean().item()}")

print("\n===使用print_gradient函数打印没有快捷连接模型的梯度")
print_gradient(model_without_shortcut, sample_input)
# layers.0.0.weight has gradient name of 0.00020173587836325169
# layers.1.0.weight has gradient name of 0.0001201116101583466
# layers.2.0.weight has gradient name of 0.0007152041653171182
# layers.3.0.weight has gradient name of 0.001398873864673078
# layers.4.0.weight has gradient name of 0.005049646366387606

# 实例化一个包含快捷连接的模型， 并观察它的比较结果
torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
print("\n===使用print_gradient函数打印基于快捷连接的模型的梯度")
print_gradient(model_with_shortcut, sample_input)
# layers.0.0.weight has gradient name of 0.22169792652130127
# layers.1.0.weight has gradient name of 0.20694106817245483
# layers.2.0.weight has gradient name of 0.32896995544433594
# layers.3.0.weight has gradient name of 0.2665732502937317
# layers.4.0.weight has gradient name of 1.3258541822433472