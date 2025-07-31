import pandas as pd

# 读取 CSV 文件
file_path = "C:/Users/admin/Desktop/IntrusionDetection/cicddos2019_dataset.csv"  # 替换为你的文件路径
df = pd.read_csv(file_path)

# 统计 Label 列中各类别的数量
label_counts = df['Label'].value_counts()

# 打印统计结果
print(label_counts)

# import pandas as pd
#
# # 读取 CSV 文件
# file_path = "C:/Users/admin/Desktop/IntrusionDetection/cicddos2019_dataset.csv"  # 替换为你的实际文件路径
# df = pd.read_csv(file_path)
#
# # 需要随机抽取的四种类别
# major_classes = ["DrDoS_NTP", "TFTP", "Benign", "Syn"]
# sample_size = 140000  # 目标样本数
#
# # 计算每个类别的采样比例
# total_major_samples = df[df['Label'].isin(major_classes)].shape[0]
# sampling_ratios = {cls: len(df[df['Label'] == cls]) / total_major_samples for cls in major_classes}
#
# # 按比例抽取数据
# sampled_data = pd.concat([
#     df[df['Label'] == cls].sample(n=int(sample_size * sampling_ratios[cls]), random_state=42, replace=False)
#     for cls in major_classes
# ])
#
# # 获取其余 14 种类别的数据
# remaining_data = df[~df['Label'].isin(major_classes)]
#
# # 合并数据
# new_dataset = pd.concat([sampled_data, remaining_data])
#
# # 保存到新文件
# new_dataset.to_csv("C:/Users/admin/Desktop/IntrusionDetection/new_cicddos2019_dataset.csv", index=False)
#
# # 打印数据集大小
# print("New dataset saved with shape:", new_dataset.shape)

# 读取 CSV 文件
file_path = "C:/Users/admin/Desktop/IntrusionDetection/new_cicddos2019_dataset.csv"  # 替换为你的文件路径
df = pd.read_csv(file_path)

# 统计 Label 列中各类别的数量
label_counts = df['Label'].value_counts()
print(df.shape)
# 打印统计结果
print(label_counts)