import os
import tiktoken
from pathlib import Path

tiktoken_cache_dir = str(Path("..", "tiktoken", ".tiktoken"))
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

print("\n\n=== 使用BPE分词器 ")
# 获取BPE分词器
bpeTokenizer = tiktoken.get_encoding("gpt2")
rawText = (
    "hello, do you like tea? <|endoftext|> in the sunlit terraces"
    "of sumunknownPlace."
)

tokenIds = bpeTokenizer.encode(rawText, allowed_special={"<|endoftext|>"})
print(tokenIds)

# 把tokenId转为字符串 -- 解码
print("\n\n===把tokenId转为字符串 -- 解码")
decodeResult = bpeTokenizer.decode(tokenIds)
print(decodeResult)