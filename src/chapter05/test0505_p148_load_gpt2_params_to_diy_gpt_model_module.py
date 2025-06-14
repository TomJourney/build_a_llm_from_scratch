import torch.nn
import numpy as np


# 把right张量返回为可训练的PyTorch参数
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch left.shape = {left.shape}, right.shape = {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

# 把params字典中的权重加载到 DiyGPTModel 实例中
# 把模型的位置信息和词元嵌入权重设置为params中指定的值
def load_weights_into_gpt(diy_gpt_model, params):
    diy_gpt_model.position_emb.weights = assign(diy_gpt_model.position_emb.weight, params['wpe'])
    diy_gpt_model.token_emb.weights = assign(diy_gpt_model.token_emb.weight, params['wte'])

    for b_index in range(len(params["blocks"])):
        # 1 使用gpt模型参数params中注意力的注意力权重替换diy_gpt_model模型对应参数
        q_w, k_w, v_w = np.split(
            (params["blocks"][b_index]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        diy_gpt_model.transformer_blocks[b_index].attention.W_query.weight = assign(
            diy_gpt_model.transformer_blocks[b_index].attention.W_query.weight, q_w.T
        )
        diy_gpt_model.transformer_blocks[b_index].attention.W_key.weight = assign(
            diy_gpt_model.transformer_blocks[b_index].attention.W_key.weight, k_w.T
        )
        diy_gpt_model.transformer_blocks[b_index].attention.W_value.weight = assign(
            diy_gpt_model.transformer_blocks[b_index].attention.W_value.weight, v_w.T
        )

        # 2 使用gpt模型参数params中注意力的偏置权重替换diy_gpt_model模型对应参数
        q_b, k_b, v_b = np.split(
            (params["blocks"][b_index]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        diy_gpt_model.transformer_blocks[b_index].attention.W_query.bias = assign(
            diy_gpt_model.transformer_blocks[b_index].attention.W_query.bias, q_b
        )
        diy_gpt_model.transformer_blocks[b_index].attention.W_key.bias = assign(
            diy_gpt_model.transformer_blocks[b_index].attention.W_key.bias, k_b
        )
        diy_gpt_model.transformer_blocks[b_index].attention.W_value.bias = assign(
            diy_gpt_model.transformer_blocks[b_index].attention.W_value.bias, v_b
        )

        # 3 使用gpt模型参数params中注意力的投影层权重替换diy_gpt_model模型对应参数
        diy_gpt_model.transformer_blocks[b_index].attention.out_proj.weight = assign(
            diy_gpt_model.transformer_blocks[b_index].attention.out_proj.weight,
            params["blocks"][b_index]["attn"]["c_proj"]["w"].T
        )
        diy_gpt_model.transformer_blocks[b_index].attention.out_proj.bias = assign(
            diy_gpt_model.transformer_blocks[b_index].attention.out_proj.bias,
            params["blocks"][b_index]["attn"]["c_proj"]["b"]
        )
        diy_gpt_model.transformer_blocks[b_index].feed_forward.layers[0].weight = assign(
            diy_gpt_model.transformer_blocks[b_index].feed_forward.layers[0].weight,
            params["blocks"][b_index]["mlp"]["c_fc"]["w"].T
        )

        # 4 使用gpt模型参数params中权重替换diy_gpt_model模型对应参数
        diy_gpt_model.transformer_blocks[b_index].feed_forward.layers[0].bias = assign(
            diy_gpt_model.transformer_blocks[b_index].feed_forward.layers[0].bias,
            params["blocks"][b_index]["mlp"]["c_fc"]["b"]
        )
        diy_gpt_model.transformer_blocks[b_index].feed_forward.layers[2].weight = assign(
            diy_gpt_model.transformer_blocks[b_index].feed_forward.layers[2].weight,
            params["blocks"][b_index]["mlp"]["c_proj"]["w"].T
        )
        diy_gpt_model.transformer_blocks[b_index].feed_forward.layers[2].bias = assign(
            diy_gpt_model.transformer_blocks[b_index].feed_forward.layers[2].bias,
            params["blocks"][b_index]["mlp"]["c_proj"]["b"]
        )

        # 5 使用gpt模型参数params中归一化参数替换diy_gpt_model模型对应参数
        diy_gpt_model.transformer_blocks[b_index].layer_norm1.scale = assign(
            diy_gpt_model.transformer_blocks[b_index].layer_norm1.scale,
            params["blocks"][b_index]["ln_1"]["g"]
        )
        diy_gpt_model.transformer_blocks[b_index].layer_norm1.shift = assign(
            diy_gpt_model.transformer_blocks[b_index].layer_norm1.shift,
            params["blocks"][b_index]["ln_1"]["b"]
        )
        diy_gpt_model.transformer_blocks[b_index].layer_norm2.scale = assign(
            diy_gpt_model.transformer_blocks[b_index].layer_norm2.scale,
            params["blocks"][b_index]["ln_2"]["g"]
        )
        diy_gpt_model.transformer_blocks[b_index].layer_norm2.shift = assign(
            diy_gpt_model.transformer_blocks[b_index].layer_norm2.shift,
            params["blocks"][b_index]["ln_2"]["b"]
        )
        diy_gpt_model.final_norm.scale = assign(diy_gpt_model.final_norm.scale, params["g"])
        diy_gpt_model.final_norm.shift = assign(diy_gpt_model.final_norm.shift, params["b"])
        diy_gpt_model.out_head.weight = assign(diy_gpt_model.out_head.weight, params["wte"])
