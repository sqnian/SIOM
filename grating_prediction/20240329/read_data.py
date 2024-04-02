import scipy.io as scio
import pandas as pd
 
# 读取 MATLAB 文件（.m）并获得变量名称列表
data = scio.loadmat('list2nain0319.mat')
# var_names = list(data.keys())
 
# print(var_names)
# 创建空白 DataFrame
df = pd.DataFrame(data["list"])
 
# 遍历每个变量名称，提取对应的值并添加到 DataFrame 中
# for var in var_names:
#     if var == "list":
#         # df[var] = data[var]

# 保存为 CSV 文件
df.to_csv('m2csv_origin.csv', index=False)

# 载入CSV文件为DataFrame
df = pd.read_csv('m2csv_origin.csv')
# print(df[-4:])
# 删除包含0值的列
non_zero_columns = df.columns[(df[:-4] != 0).any()]

# print(non_zero_columns)
# 选择非零值的列
df = df[non_zero_columns].T

# 随机排序数据
df_shuffled = df.sample(frac=1).reset_index(drop=True)

# # 将更改后的DataFrame保存回CSV文件
df_shuffled.to_csv('m2csv_result.csv', index=False)