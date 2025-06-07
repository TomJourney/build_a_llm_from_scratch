import torch.nn as nn

from src.chapter03.test0306_p78_multi_head_attention_module import MultiHeadAttention
from src.chapter04.test0402_p90_layer_norm_module import LayerNorm
from src.chapter04.test0403_p96_feed_forward_module import FeedForward

# 定义Transformer块类
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.feed_forward = FeedForward(cfg)
        self.layer_norm1 = LayerNorm(cfg["emb_dim"])
        self.layer_norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 在注意力块中添加快捷连接
        shortcut = x
        # 1. 层归一化
        x = self.layer_norm1(x)
        # 2. 多头注意力机制
        x = self.attention(x)
        # 3. 丢弃部分权值，防止过拟合
        x = self.drop_shortcut(x)
        # 把原始输入添加回来
        x = x + shortcut

        # 在前馈层中添加快捷连接
        shortcut = x
        # 1. 层归一化
        x = self.layer_norm2(x)
        # 2. 前馈神经网络
        x = self.feed_forward(x)
        # 3. 丢弃部分权值，防止过拟合
        x = self.drop_shortcut(x)
        # 把原始输入添加回来
        x = x + shortcut
        return x
