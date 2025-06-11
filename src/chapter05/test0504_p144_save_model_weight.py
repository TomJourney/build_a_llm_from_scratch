import torch

from src.chapter04.test0406_p107_gpt_model_module import DiyGPTModel

print("\n\n===使用python字典指定gpt模型的配置")
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

# 实例化gpt模型（使用自定义GPT模型）
torch.manual_seed(123)
# 创建自定义gpt模型实例
diy_gpt_model = DiyGPTModel(GPT_CONFIG_124M)
# 获取设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n\n=== step1：保存模型权重，推荐使用torch.save保存模型的state_dict")
torch.save(diy_gpt_model.state_dict(), "diy_gpt_model_weights.pth")
# diy_gpt_model.pth是保存state_dict的文件名，其中 pth是PyTorch文件的扩展名

print("\n\n=== step2：加载模型权重，使用torch.load()")
loaded_diy_gpt_model = DiyGPTModel(GPT_CONFIG_124M)
loaded_diy_gpt_model.load_state_dict(torch.load("diy_gpt_model_weights.pth", map_location=device))
# loaded_diy_gpt_model.eval() 把模型切换为推断模式 ，这会禁用模型的dropout层
loaded_diy_gpt_model.eval()

# 像 AdamW这样的自适应优化器可以为每个模型权重存储额外的参数
# AdamW可以使用历史数据动态调整每个模型参数的学习率，如果没有它，优化器就会被重置，模型学习效果可能不佳

# 使用AdamW优化器和train_model_simple函数，对模型实例diy_gpt_model进行10轮训练
# .parameters()方法返回模型的所有可训练权重参数
optimizer = torch.optim.AdamW(
    diy_gpt_model.parameters(), lr=0.0004, weight_decay=0.1)
print("\n\n=== step3：使用torch.save()保存模型和优化器的state_dict内容")
torch.save({
        "model_state_dict": diy_gpt_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    },
    "model_and_optimizer_weights.pth"
)


print("\n\n=== step4：使用torch.load()加载保存的数据，再使用 load_state_dict恢复模型和优化器的状态")
checkpoint = torch.load("model_and_optimizer_weights.pth", map_location=device)
diy_gpt_model2 = DiyGPTModel(GPT_CONFIG_124M)
# 恢复模型权重
diy_gpt_model2.load_state_dict(checkpoint["model_state_dict"])
# 创建优化器
loaded_optimizer = torch.optim.AdamW(diy_gpt_model2.parameters(), lr=5e-4, weight_decay=0.1)
# 恢复优化器权重
loaded_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
diy_gpt_model2.train()


