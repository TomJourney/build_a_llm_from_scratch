import pandas as pd
from sympy.printing.pytorch import torch
from torch.utils.data import Dataset


# 垃圾邮件数据集类-DiySpamDataset
# 垃圾邮件数据集类处理几个关键任务： 1-把文本消息编码为词元序列，2-识别训练数据集中最长的序列，
# 3-确保所有其他序列都使用填充词元进行填充，以匹配最长序列的长度
class DiySpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        # 文本分词
        self.encoded_text_array = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

            # 如果序列长度超过 max_length，则进行截断
            self.encoded_text_array = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_text_array
            ]

        # 填充到最长序列的长度
        self.encoded_text_array = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_text_array
        ]

    def __getitem__(self, index):
        encoded = self.encoded_text_array[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    # 识别数据集中数据序列的最大长度
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_text_array:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
