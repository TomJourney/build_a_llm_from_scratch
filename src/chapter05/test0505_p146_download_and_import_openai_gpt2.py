import urllib.request
from gpt_download import download_and_load_gpt2

# 下载加载gpt-2架构设置与权重参数的python模块（本书作者提供）
# print("\n step1：下载gpt-download.py")
# url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch05/01_main-chapter-code/gpt_download.py"
# filename = url.split("/")[-1]
# urllib.request.urlretrieve(url, filename)

# 从gpt-download.py中 导入 download_and_load_gpt2函数
print("\n step2：使用 download_and_load_gpt2函数 加载gpt-2架构设置和权重参数到python会话中")
gpt2_settings, gpt2_params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2", is_download=False
)
# 执行上述代码-download_and_load_gpt2 函数， 将下载参数量为1.24亿的GPT-2模型的7个文件
print("\n=== 执行上述代码-download_and_load_gpt2函数， 将下载参数量为1.24亿的GPT-2模型的7个文件，， 下载完成")

# 打印download_and_load_gpt2函数 加载gpt-2架构设置和权重参数
print("gpt2_settings = ", gpt2_settings)
print("gpt2_params = ", gpt2_params)
# settings =  {'n_vocab': 50257, 'n_ctx': 1024, 'n_embd': 768, 'n_head': 12, 'n_layer': 12}
# params =  {'blocks': [{'attn': {'c_attn': {'b': array([ 0.48033914, -0.5254326 , -0.42926455, ...,  0.01257301,
#        -0.04987717,  0.00324764], dtype=float32), 'w': array([[-0.4738484 , -0.26136586, -0.09780374, ...,  0.05132535,
#         -0.0584389 ,  0.02499568] .....

print("gpt2_params.keys() = ", gpt2_params.keys())
print("gpt2_params[\"wte\"]", gpt2_params["wte"])
# gpt2_params.keys() =  dict_keys(['blocks', 'b', 'g', 'wpe', 'wte'])
# gpt2_params["wte"] [[-0.11010301 -0.03926672  0.03310751 ... -0.1363697   0.01506208
#    0.04531523]
#  [ 0.04034033 -0.04861503  0.04624869 ...  0.08605453  0.00253983
#    0.04318958]
#  [-0.12746179  0.04793796  0.18410145 ...  0.08991534 -0.12972379
#   -0.08785918]
#  ...
#  [-0.04453601 -0.05483596  0.01225674 ...  0.10435229  0.09783269
#   -0.06952604]
#  [ 0.1860082   0.01665728  0.04611587 ... -0.09625227  0.07847701
#   -0.02245961]
#  [ 0.05135201 -0.02768905  0.0499369  ...  0.00704835  0.15519823
#    0.12067825]]