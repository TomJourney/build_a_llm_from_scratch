import json

from src.chapter07.test0702_p189_format_input_to_alpaca_module import format_input_to_alpaca

# 读取数据集
with open('./dataset/instruction-data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

print("\n=== 划分数据集： 训练集85%， 测试集10%， 验证集5%")
train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
validate_portion = int(len(data) * 0.05)

train_data = data[:train_portion]
test_data = data[train_portion:train_portion+test_portion]
validate_data = data[train_portion+test_portion:]

print("训练集长度 = ", len(train_data))
print("测试集长度 = ", len(test_data))
print("验证集长度 = ", len(validate_data))
# 训练集长度 =  935
# 测试集长度 =  110
# 验证集长度 =  55