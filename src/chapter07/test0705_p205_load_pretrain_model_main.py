from src.chapter04.test0406_p107_gpt_model_module import DiyGPTModel
from src.chapter05.gpt_download import download_and_load_gpt2
from src.chapter05.test0505_p148_load_gpt2_params_to_diy_gpt_model_module import load_weights_into_gpt

# 测试案例-加载预训练模型

# 测试用例-计算分类准确率
# 【1】模型配置信息
# 基本配置，包括词汇表大小， 上下文长度， dropout率-丢弃率， 查询-键-值的偏置
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}
# 模型参数配置
# 字典保存不同模型尺寸的GPT模型参数
gpt2_model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (744M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25}
}
# 选择参数量为3.55亿的模型
CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(gpt2_model_configs[CHOOSE_MODEL])

# 解析模型的参数大小
pretrain_model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
print("pretrain_model_size = ", pretrain_model_size)  # 355M

# 下载模型
settings, params = download_and_load_gpt2(model_size=pretrain_model_size, models_dir="gpt2", is_download=True)

# 创建大模型实例，加载权重到模型实例
gpt2_355_model = DiyGPTModel(BASE_CONFIG)
load_weights_into_gpt(gpt2_355_model, params)
# 设置大模型为评估模式
gpt2_355_model.eval()

# checkpoint: 100%|██████████| 77.0/77.0 [00:00<00:00, 38.6kiB/s]
# encoder.json: 100%|██████████| 1.04M/1.04M [00:00<00:00, 1.06MiB/s]
# hparams.json: 100%|██████████| 91.0/91.0 [00:00<00:00, 879iB/s]
# model.ckpt.data-00000-of-00001: 100%|██████████| 1.42G/1.42G [01:17<00:00, 18.3MiB/s]
# model.ckpt.index: 100%|██████████| 10.4k/10.4k [00:00<00:00, 5.24MiB/s]
# model.ckpt.meta: 100%|██████████| 927k/927k [00:00<00:00, 1.02MiB/s]
# vocab.bpe: 100%|██████████| 456k/456k [00:01<00:00, 283kiB/s]

# 验证加载的大模型

