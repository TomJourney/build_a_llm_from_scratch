import torch


# 基于温度缩放与Top-k采样修改文本生成函数-generate_text_simple()
# 生成文本（index_array是当前文本的索引数组，形状为(batch, n_tokens)）
def based_temperature_topk_generate_text_simple(gpt_model, index_array, max_new_tokens, context_size, temperature=0.0,
                                                top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        # 把当前文本截断至支持的长度。若大模型仅支持5个词元，但输入文本长度为10，则只有最后5个词元被用作输入文本
        sub_input_index_array = index_array[:, -context_size:]
        with torch.no_grad():
            logits = gpt_model(sub_input_index_array)

        # 只关注最后一个输出的内容，因为形状会从 (batch, n_token, vocab_size) 变为 (batch, vocab_size)
        logits = logits[:, -1, :]
        print("\n logits.shape = ", logits.shape)
        # logits.shape = torch.Size([1, 50257])

        # 使用top-k采样筛选logits
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            # top_logits.shape = torch.Size([1, 50])

            min_value = top_logits[:, -1]
            logits = torch.where(
                logits < min_value,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        # 使用温度缩放
        if temperature > 0.0:
            logits = logits / temperature
            probabilities = torch.softmax(logits, dim=-1)
            index_next = torch.multinomial(probabilities, num_samples=1)
        # 当禁用温度缩放时，则执行贪心解码，选取下一个词元
        else:
            index_next = torch.argmax(logits, dim=-1, keepdim=True)
        if index_next == eos_id:
            break
        index_array = torch.cat((index_array, index_next), dim=1)
    return index_array
