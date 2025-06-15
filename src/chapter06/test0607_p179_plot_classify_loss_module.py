# 绘制分类损失曲线
import matplotlib.pyplot as plt


def plot_values(epoch_seen, examples_seen, train_values, validate_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # 绘制训练集损失和验证集损失与轮数的关系
    ax1.plot(epoch_seen, train_values, label=f"training {label}")
    ax1.plot(epoch_seen, validate_values, linestyle="-.", label=f"validate {label}")

    ax1.set_xlabel("epochs-轮次")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # 为所见样本画第2个x轴
    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlable("examples seen")

    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()
