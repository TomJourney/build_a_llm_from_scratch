from pathlib import Path

import tiktoken

from src.chapter06.test0603_p1160_spam_dataset_module import DiySpamDataset

# 获取tiktoken中的gpt2分词器
gpt2_tokenizer = tiktoken.get_encoding("gpt2")
# 把<|endoftext|>作为填充词元，词元<|endoftext|>的词元id等于50256
print(gpt2_tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
# [50256]

train_dataset = DiySpamDataset(
    csv_file=Path("dataset") / "train.csv",
    max_length=None,
    tokenizer=gpt2_tokenizer
)


