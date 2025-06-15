import torch


# 使用微调后的模型对文本分类
def classify_review(text, diy_gpt_model, tokenizer, device, max_length=None, pad_token_id=50256):
    diy_gpt_model.eval()

    # 准备模型的输入数据
    input_ids = tokenizer.encode(text)
    supprted_context_length = diy_gpt_model.position_emb.weight.shape[0]

    # 截断过长的序列
    input_ids = input_ids[:min(max_length, supprted_context_length)]

    # 填充序列至最长序列长度
    input_ids += [pad_token_id] * (max_length - len(input_ids))

    # 添加批次维度
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    # 推理时，不需要计算梯度
    with torch.no_grad():
        # 最后一个输出词元的logits
        logits = diy_gpt_model(input_tensor)[:, -1, :]

    predicted_label = torch.argmax(logits, dim=-1).item()
    # 返回分类结果
    return "spam" if predicted_label == 1 else "not spam"
