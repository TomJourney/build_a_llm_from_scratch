import tiktoken_ext.openai_public
import inspect

print(dir(tiktoken_ext.openai_public))
# The encoder we want is cl100k_base, we see this as a possible function

print(inspect.getsource(tiktoken_ext.openai_public.gpt2))
# The URL should be in the 'load_tiktoken_bpe function call'
