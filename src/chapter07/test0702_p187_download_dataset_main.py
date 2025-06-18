import json

# 下载并打印数据集
# url=https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json

with open('./dataset/instruction-data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

print("数据量 = ", len(data))
# 数据量 =  1100
# 打印数据项=data[50]
print("data[50] = ", data[50])
# data[50] =  {'instruction': 'Identify the correct spelling of the following word.', 'input': 'Ocassion', 'output': "The correct spelling is 'Occasion.'"}

# {
# 	"instruction": "Identify the correct spelling of the following word.",
# 	"input": "Ocassion",
# 	"output": "The correct spelling is Occasion."
# }

# 打印数据项=data[999]
print("data[999] = ", data[999])
# data[999] =  {'instruction': "What is an antonym of 'complicated'?", 'input': '', 'output': "An antonym of 'complicated' is 'simple'."}

# {
# 	"instruction": "What is an antonym of 'complicated'?",
# 	"input": "",
# 	"output": "An antonym of "complicated" is 'simple'."
# }