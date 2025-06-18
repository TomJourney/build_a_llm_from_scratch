import torch


# 自定义聚合函数-版本1
def custom_agg_function_v1(
        batch, pad_token_id=50256, device="cpu"
):
    # 找到批次中最长的序列
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_1st = []

    # 填充并准备输入
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
                new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )
        # 删除之前添加的额外填充词元
        inputs = torch.tensor(padded[:-1])
        inputs_1st.append(inputs)

    # 输入列表变成一个张量并转移到目标设备
    inputs_tensor = torch.stack(inputs_1st).to(device)
    return inputs_tensor


# 自定义聚合函数-版本2
def custom_agg_function_v2(
        batch, pad_token_id=50256, ignore_index=-100, allow_max_length=None, device="cpu"
):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_1st, targets_1st = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        # 将序列填充至max_length
        padded = (
                new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )

        # 截断输入的最后一个词元
        inputs = torch.tensor(padded[:-1])
        # 向左移动一个位置得到目标
        targets = torch.tensor(padded[1:])

        # 目标序列中的除第1个填充词元外的所有填充词元全部替换为 ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # 可选地截断至最大序列长度
        if allow_max_length is not None:
            inputs = inputs[:allow_max_length]
            targets = targets[:allow_max_length]

        inputs_1st.append(inputs)
        targets_1st.append(targets)

    inputs_tensor = torch.stack(inputs_1st).to(device)
    targets_tensor = torch.stack(targets_1st).to(device)
    return inputs_tensor, targets_tensor
