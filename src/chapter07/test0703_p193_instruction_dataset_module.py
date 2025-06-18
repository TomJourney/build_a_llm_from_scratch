import torch
from torch.utils.data import Dataset

from src.chapter07.test0702_p189_format_input_to_alpaca_module import format_input_to_alpaca

# 指令微调的数据集类
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            # 预先词元化文本
            instruction_plus_input = format_input_to_alpaca(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)