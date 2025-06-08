import torch


# 生成文本（index_array是当前文本的索引数组，形状为(batch, n_tokens)）
def generate_text_simple(gpt_model, index_array, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        # 把当前文本截断至支持的长度。若大模型仅支持5个词元，但输入文本长度为10，则只有最后5个词元被用作输入文本
        sub_input_index_array = index_array[:, -context_size:]
        with torch.no_grad():
            logits = gpt_model(sub_input_index_array)

        # 只关注最后一个输出的内容，因为形状会从 (batch, n_token, vocab_size) 变为 (batch, vocab_size)
        logits = logits[:, -1, :]
        # 使用softmax函数把logits转为概率分布
        probability_distribution = torch.softmax(logits, dim=-1)
        # 确定最大概率的位置，该位置就是预测的下一个词元id
        index_next = torch.argmax(probability_distribution, dim=-1, keepdim=True)
        # 把计算出的下一个词元的索引（词元id）追加到索引数组中，index_array会变为(batch, n_tokens+1)
        index_array = torch.cat((index_array, index_next), dim=1)
    return index_array
