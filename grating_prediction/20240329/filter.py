import pandas as pd

# 载入CSV文件为DataFrame
df = pd.read_csv('m2csv_result.csv')
# print(df[df[df.columns[-1]] == 1.0])
# 获取最后一列的列名
last_column_name = df.columns[-1]

# 筛选最后一列值为1的所有行
filtered_df = df[df[last_column_name] == 1]

print(filtered_df)
# # # 将更改后的DataFrame保存回CSV文件
filtered_df.to_csv('m2csv_filter_result.csv', index=False)