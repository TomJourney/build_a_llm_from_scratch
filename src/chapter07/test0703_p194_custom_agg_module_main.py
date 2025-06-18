from src.chapter07.test0703_p194_custom_agg_module import custom_agg_function_v1, custom_agg_function_v2

inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]

batch = (
    inputs_1,
    inputs_2,
    inputs_3
)
# 填充批次中的样本使得样本的长度相等
print("\n=== 填充批次中的样本使得样本的长度相等")
print(custom_agg_function_v1(batch))
# tensor([[    0,     1,     2,     3,     4],
#         [    5,     6, 50256, 50256, 50256],
#         [    7,     8,     9, 50256, 50256]])

print("\n=== 调用自定义聚合函数-版本2")
inputs, targets = custom_agg_function_v2(batch)
print("inputs = ", inputs)
print("targets = ", targets)
# inputs =  tensor([[    0,     1,     2,     3,     4],
#         [    5,     6, 50256, 50256, 50256],
#         [    7,     8,     9, 50256, 50256]])
# targets =  tensor([[    1,     2,     3,     4, 50256],
#         [    6, 50256,  -100,  -100,  -100],
#         [    8,     9, 50256,  -100,  -100]])
