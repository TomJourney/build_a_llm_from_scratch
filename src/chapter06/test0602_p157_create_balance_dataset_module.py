import pandas as pd


def create_balanced_dataset(df):
    # 统计垃圾消息的样本数量
    num_spam = df[df["Label"] == "spam"].shape[0]

    # 随机采样非垃圾消息，使其数量与垃圾消息一致
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state=123
    )

    # 将垃圾消息与采样后的垃圾消息组合，构成平衡数据集
    balanced_df = pd.concat([
        ham_subset, df[df["Label"] == "spam"]
    ])
    return balanced_df
