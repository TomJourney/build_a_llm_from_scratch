import urllib.request
from gpt_download import download_and_load_gpt2

# 下载OpenAI通过TensorFlow保存的GPT-2模型权重
# print("\n step1：下载gpt-download.py")
# url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch05/01_main-chapter-code/gpt_download.py"
# filename = url.split("/")[-1]
# urllib.request.urlretrieve(url, filename)

# 从gpt-download.py中 导入 download_and_load_gpt2函数
print("\n step2：使用 download_and_load_gpt2函数 加载gpt-2架构设置和权重参数到python会话中")
settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
)
# 执行上述代码-download_and_load_gpt2 函数， 将下载参数量为1.24亿的GPT-2模型的7个文件
print("\n=== 执行上述代码-download_and_load_gpt2函数， 将下载参数量为1.24亿的GPT-2模型的7个文件，， 下载完成")

