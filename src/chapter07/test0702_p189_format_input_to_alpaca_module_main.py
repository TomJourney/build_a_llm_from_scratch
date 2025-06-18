import json

from src.chapter07.test0702_p189_format_input_to_alpaca_module import format_input_to_alpaca

# 读取数据集
with open('./dataset/instruction-data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

print("\n=== 调用 format_input_to_alpaca 把数据data[50]转为 Alpaca提示词格式")
model_input = format_input_to_alpaca(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"
print(model_input + desired_response)

# Below is an instruction that describes a task.Write a response that appropriately completes the request.
#
# ### Instruction:
# Identify the correct spelling of the following word.
#
# ### Input:
# Ocassion
#
# ### Response:
# The correct spelling is 'Occasion.'

print("\n=== 调用 format_input_to_alpaca 把数据data[999]转为 Alpaca提示词格式")
model_input = format_input_to_alpaca(data[999])
desired_response = f"\n\n### Response:\n{data[999]['output']}"
print(model_input + desired_response)

# Below is an instruction that describes a task.Write a response that appropriately completes the request.
#
# ### Instruction:
# What is an antonym of 'complicated'?
#
# ### Response:
# An antonym of 'complicated' is 'simple'.