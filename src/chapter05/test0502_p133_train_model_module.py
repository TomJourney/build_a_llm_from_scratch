import torch

from src.chapter04.test0406_p112_gpt_model_generate_text_module import generate_text_simple
from src.chapter05.test0501_p119_text_to_token_transfer_util_module import text_to_tokens_ids, token_ids_to_text
from src.chapter05.test0501_p127_compute_train_test_loss_module import compute_loss_batch, compute_loss_loader


# 打印训练集与测试集的损失
def evaluate_model(diy_gpt_model, train_loader, test_loader, device, eval_iter):
    diy_gpt_model.eval()
    with torch.no_grad():
        train_loss = compute_loss_loader(train_loader, diy_gpt_model, device, num_batches=eval_iter)
        validate_loss = compute_loss_loader(test_loader, diy_gpt_model, device, num_batches=eval_iter)
    diy_gpt_model.train()
    return train_loss, validate_loss


# 生成文本并打印样本
def generate_and_print_sample(diy_gpt_model, tokenizer, device, start_context):
    diy_gpt_model.eval()
    context_size = diy_gpt_model.position_emb.weight.shape[0]
    encoded = text_to_tokens_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(gpt_model=diy_gpt_model, index_array=encoded, max_new_tokens=50,
                                         context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print("decoded_text = ", decoded_text.replace("\n", " "))
    diy_gpt_model.train()


# 训练模型
def train_model_simple(diy_gpt_model, train_loader, validate_loader, optimizer, device, num_epochs, eval_frequency,
                       eval_iter, start_context, tokenizer):
    train_losses, validate_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        diy_gpt_model.train()
        for input_batch, target_batch in train_loader:
            # 重置上一个批次迭代中的损失梯度
            optimizer.zero_grad()
            loss = compute_loss_batch(input_batch, target_batch, diy_gpt_model, device)

            # 计算损失梯度
            loss.backward()
            # 使用损失梯度更新模型权重
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            # 可选的评估步骤
            if global_step % eval_frequency == 0:
                train_loss, validate_loss = evaluate_model(diy_gpt_model, train_loader, validate_loader, device, eval_iter)
                train_losses.append(train_loss)
                validate_losses.append(validate_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (step {global_step:06d}) :"
                      f"train_loss = {train_loss:.3f}, "
                      f"validate_loss = {validate_loss:.3f}")
        generate_and_print_sample(diy_gpt_model, tokenizer, device, start_context)
    return train_losses, validate_losses, track_tokens_seen
