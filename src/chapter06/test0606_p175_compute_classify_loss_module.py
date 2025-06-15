import torch

# 计算分类损失函数
def compute_classify_loss_batch(input_batch, target_batch, diy_gpt_model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    # 最后一个输出词元的logits
    logits = diy_gpt_model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

# 计算损失加载器
def compute_classify_loss_loader(data_loader, diy_gpt_model, device, num_batches=None):
    total_loss = 0;
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = compute_classify_loss_batch(
                input_batch, target_batch, diy_gpt_model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches
