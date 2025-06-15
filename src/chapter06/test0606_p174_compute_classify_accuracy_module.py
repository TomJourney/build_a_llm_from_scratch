import torch


# 计算分类准确率加载器
def compute_accuracy_loader(data_loader, diy_gpt_model, device, num_batches=None):
    diy_gpt_model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            # 最后一个输出词元的logits
            with torch.no_grad():
                logits = diy_gpt_model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (
                (predicted_labels == target_batch).sum().item()
            )
        else:
            break
        return correct_predictions / num_examples
