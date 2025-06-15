# train_fraction 训练集比例
# validation_fraction 验证集比例
def random_split(df, train_fraction, validation_fraction):
    # 打乱整个 Dataframe
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # 计算拆分索引
    train_end = int(len(df) * train_fraction)
    validation_end = train_end + int(len(df) * validation_fraction)

    # 拆分 Dataframe，包括训练集，验证集， 测试集
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df
