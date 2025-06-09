import torch

# 计算训练集加载器与验证集加载器返回的给定批次的交叉熵损失函数
def compute_loss_batch(input_batch, target_batch, diy_gpt_model, device):
    # to(device) 可以把输入数据转移到GPU上
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = diy_gpt_model(input_batch)
    cross_entropy_loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return cross_entropy_loss

# 交叉熵损失计算加载器
def compute_loss_loader(data_loader, diy_gpt_model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = compute_loss_batch(input_batch, target_batch, diy_gpt_model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches