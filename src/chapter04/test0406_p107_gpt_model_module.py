import torch
import torch.nn as nn
from src.chapter04.test0405_p103_transformer_block_module import TransformerBlock
from src.chapter04.test0402_p90_layer_norm_module import LayerNorm
class DiyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.position_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout_emb = nn.Dropout(cfg["drop_rate"])

        # Transformer块的顺序栈，其层数与cfg指定的层数相同
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        # 层归一化
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 无偏置的线性输出头，把Transformer的输出投射到分词器的词汇空间，为词汇中的每个词元生成分数
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    # 前向传播
    def forward(self, in_index):
        batch_size, sequence_length = in_index.shape
        token_embs = self.token_emb(in_index)

        # device的设置允许我们在CPU或GPU上训练模型，具体取决于输入数据所在设备
        position_embs = self.position_emb(
            torch.arange(sequence_length, device=in_index.device)
        )
        x = token_embs + position_embs
        x = self.dropout_emb(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        # logits表示下一个词元的非归一化概率
        logits = self.out_head(x)
        return logits
