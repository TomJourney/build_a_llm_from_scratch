import torch

# 微调模型
from src.chapter06.test0606_p175_compute_classify_loss_module import compute_classify_loss_batch, \
    compute_classify_loss_loader
from src.chapter06.test0606_p174_compute_classify_accuracy_module import compute_accuracy_loader

# 定义评估模型函数
def evaluate_model(diy_gpt_model, train_loader, validate_loader, device, eval_iter):
    diy_gpt_model.eval()
    with torch.no_grad():
        train_loss = compute_classify_loss_loader(
            train_loader, diy_gpt_model, device, num_batches=eval_iter
        )
        validate_loss = compute_classify_loss_loader(
            validate_loader, diy_gpt_model, device, num_batches=eval_iter
        )
    diy_gpt_model.train()
    return train_loss, validate_loss


# 定义训练分类器
def train_classifier_simple(
        diy_gpt_model, train_loader, validate_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    # 初始化列表， 以跟踪损失和所见样本
    train_losses, validate_losses, train_accurate_array, validate_accurate_array = [], [], [], []
    examples_seen, global_step = 0, -1

    # 主训练循环
    for epoch in range(num_epochs):
        # 设置模型为训练模式
        diy_gpt_model.train()

        # 遍历每轮的批次
        for input_batch, target_batch in train_loader:
            # 重置上一个批次的损失梯度
            optimizer.zero_grad()
            loss = compute_classify_loss_batch(input_batch, target_batch, diy_gpt_model, device)
            # 计算损失梯度
            loss.backward()
            # 使用损失梯度更新模型权重
            optimizer.step()
            # 跟踪样本，跟踪训练进度
            examples_seen += input_batch.shape[0]
            global_step += 1

            # 可选的评估步骤
            if global_step % eval_freq == 0:
                train_loss, validate_loss = evaluate_model(
                    diy_gpt_model, train_loader, validate_loader, device, eval_iter)
                train_losses.append(train_loss)
                validate_losses.append(validate_loss)
                print(f"Ep {epoch + 1} step {global_step:06d}: "
                      f"train loss = {train_loss:.3f}, "
                      f"validate loss = {validate_loss:.3f}")

        # 每轮训练后计算分类准确率
        train_accuracy = compute_accuracy_loader(
            train_loader, diy_gpt_model, device, num_batches=eval_iter
        )
        validate_accuracy = compute_accuracy_loader(
            validate_loader, diy_gpt_model, device, num_batches=eval_iter
        )
        # 打印分类准确率
        print(f"train_accuracy = {train_accuracy*100:.2f}%")
        print(f"validate_accuracy = {validate_accuracy * 100:.2f}%")

        train_accurate_array.append(train_accuracy)
        validate_accurate_array.append(validate_accuracy)

    return train_losses, validate_losses, train_accurate_array, validate_accurate_array, examples_seen