import pandas as pd

# 读入两个 csv
df1 = pd.read_csv("data/merged_incidents.csv")
df2 = pd.read_csv("data/tsne_2d.csv")

# 如果第二个文件存在重复 RECORD_NO_MASTER：去重
df2 = df2.drop_duplicates(subset=["RECORD_NO_MASTER"], keep="first")

# 合并，左表是 df1，只补齐 x, y
merged = df1.merge(df2[["RECORD_NO_MASTER", "x", "y"]],
                   on="RECORD_NO_MASTER",
                   how="left")

# 保存结果
merged.to_csv("merged_incidents_tsne.csv", index=False)

print("Merged saved to merged_output.csv")
