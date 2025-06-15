from pathlib import Path

import pandas as pd

from src.chapter06.test0601_p156_download_unzip_dataset_module import download_and_unzip_spam_data

# 下载并解压数据集
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
print("下载并解压数据集完成")

# 加载数据集
data_file_path = Path("dataset") / "SMSSpamCollection.tsv"
df = pd.read_csv(
    data_file_path, sep="\t", header=None, names=["Label", "Text"]
)

print("\n\n===查看数据分布")
print(f"数据分布 = \n", df["Label"].value_counts())
#  Label
# ham     4825
# spam     747
# Name: count, dtype: int64


