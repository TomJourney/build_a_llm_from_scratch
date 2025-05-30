from importlib.metadata import version
import tiktoken

print("tiktoken version=" , version("tiktoken"))

print("\n\n=== 使用BPE分词器 ")
# 获取BPE分词器
bpeTokenizer = tiktoken.get_encoding("gpt2")
rawText = (
    "hello, do you like tea? <|endoftext|> in the sunlit terraces"
    "of sumunknownPlace."
)

tokenIds = bpeTokenizer.encode(rawText, allowed_special={"<|endoftext|>"})
print(tokenIds)
