from pathlib import Path

import pandas as pd

from src.chapter06.test0602_p157_create_balance_dataset_module import create_balanced_dataset
from src.chapter06.test0602_p158_split_dataset_module import random_split

# 下载并解压数据集
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
print("下载并解压数据集完成")

# 加载数据集
data_file_path = Path("dataset") / "SMSSpamCollection.tsv"
df = pd.read_csv(
    data_file_path, sep="\t", header=None, names=["Label", "Text"]
)

print("\n\n=== 步骤1： 查看数据分布")
print("数据分布 = \n", df["Label"].value_counts())
#  Label
# ham     4825
# spam     747
# Name: count, dtype: int64

print("\n\n=== 步骤2： 创建平衡数据集")
balanced_df = create_balanced_dataset(df)
print("balanced_df = \n", balanced_df["Label"].value_counts())
#  Label
# ham     747
# spam    747
# Name: count, dtype: int64

print("\n\n=== 步骤2-1： 把ham与spam分别转换为标签0和1")
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

print("\n\n=== 步骤3： 划分数据集，训练集70%， 验证集10%， 测试集20%； 其中总数量=747*2=1494")
train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
print("把数据集保存为csv，以便重用")
train_df.to_csv(Path("dataset") / "train.csv", index=None)
validation_df.to_csv(Path("dataset") / "validation.csv", index=None)
test_df.to_csv(Path("dataset") / "test.csv", index=None)

print("\n\n===统计各数据集的数据量")
print("训练集数据量 = ", pd.read_csv(Path("dataset")/"train.csv").shape)
print("验证集数据量 = ", pd.read_csv(Path("dataset")/"validation.csv").shape)
print("测试集数据量 = ", pd.read_csv(Path("dataset")/"test.csv").shape)
# ===统计各数据集的数据量
# 训练集数据量 =  (1045, 2)
# 验证集数据量 =  (149, 2)
# 测试集数据量 =  (300, 2)