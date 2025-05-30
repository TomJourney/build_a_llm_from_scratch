import hashlib
from pathlib import Path

blobpath = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe"
cache_key = hashlib.sha1(blobpath.encode()).hexdigest()
print(cache_key)