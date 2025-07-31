import os
from collections import Counter

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing, metrics
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data loading
# 文件路径
train_file = '../NSL-KDD/KDDTrain+.txt'
test_file = '../NSL-KDD/KDDTest+.txt'

# 列名
columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
           'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
           'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
           'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
           'num_access_files', 'num_outbound_cmds', 'is_host_login',
           'is_guest_login', 'count', 'srv_count', 'serror_rate',
           'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
           'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
           'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
           'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
           'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
           'dst_host_srv_rerror_rate', 'subclass', 'difficulty_level']

# 加载数据
df_train = pd.read_csv(train_file, header=None, names=columns)
df_test = pd.read_csv(test_file, header=None, names=columns)

# 删除 difficulty_level 列
df_train = df_train.drop(columns=['difficulty_level'])
df_test = df_test.drop(columns=['difficulty_level'])

# 合并数据
combined_data = pd.concat([df_train, df_test], ignore_index=True)

# 独热编码
def one_hot(df, cols):
    for col in cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df = pd.concat([df, dummies], axis=1).drop(columns=[col])
    return df

categorical_cols = ['protocol_type', 'service', 'flag']
combined_data = one_hot(combined_data, categorical_cols)

# 提取标签并移除 subclass 列
labels = combined_data.pop('subclass')

# 归一化
scaler = preprocessing.MinMaxScaler()
combined_data = pd.DataFrame(scaler.fit_transform(combined_data), columns=combined_data.columns)

# 标签映射
attack_map = {
    'DoS': ["apache2", "back", "land", "neptune", "mailbomb", "pod", "processtable", "smurf", "teardrop", "udpstorm",
            "worm"],
    'Probe': ["ipsweep", "mscan", "nmap", "portsweep", "saint", "satan"],
    'U2R': ["buffer_overflow", "loadmodule", "perl", "ps", "rootkit", "sqlattack", "xterm"],
    'R2L': ["ftp_write", "guess_passwd", "httptunnel", "imap", "multihop", "named", "phf", "sendmail", "Snmpgetattack",
            "spy", "snmpguess", "warezclient", "warezmaster", "xlock", "xsnoop"],
    'Normal': ["normal"]
}

label_map = {}
for i, (category, attacks) in enumerate(attack_map.items()):
    for attack in attacks:
        label_map[attack] = i

# 标签编码
labels = labels.map(label_map)

# 将检测到的空标签归为 Normal 类
normal_label = label_map['normal']  # 获取 Normal 类的标签值
labels = labels.fillna(normal_label)  # 填充空值为 Normal 标签

# 各类别数量统计
DoSCount = labels.isin([label_map[attack] for attack in attack_map['DoS']]).sum()
ProbeCount = labels.isin([label_map[attack] for attack in attack_map['Probe']]).sum()
U2RCount = labels.isin([label_map[attack] for attack in attack_map['U2R']]).sum()
R2LCount = labels.isin([label_map[attack] for attack in attack_map['R2L']]).sum()
NormalCount = labels.isin([label_map[attack] for attack in attack_map['Normal']]).sum()

print(f"DoS: {DoSCount}, Probe: {ProbeCount}, U2R: {U2RCount}, R2L: {R2LCount}, Normal: {NormalCount}")

# 检查是否有空值
print("是否有空值:", combined_data.isnull().values.any())
print("标签是否有空值:", labels.isnull().values.any())

# 转换为张量
X = torch.tensor(combined_data.values, dtype=torch.float32)
y = torch.tensor(labels.values, dtype=torch.long)

# Dataset and DataLoader
class NSL_KDD_Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Model definition
class NSLKDDModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(NSLKDDModel, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv1d(1, 32, kernel_size=32, padding='same')
        self.conv2 = nn.Conv1d(1, 32, kernel_size=64, padding='same')
        self.conv3 = nn.Conv1d(1, 32, kernel_size=96, padding='same')

        # 批量归一化（在卷积后对通道进行归一化）
        self.bn = nn.BatchNorm1d(32 * 3)

        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=4)

        # GRU（双向）
        self.gru = nn.GRU(32 * 3, 64, batch_first=True, bidirectional=True)

        # 注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        # 全连接层
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_dim // 4 * 128, num_classes)  # 根据池化和 GRU 输出形状调整
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 输入形状检查
        # print(f"Input shape before Conv1d: {x.shape}")

        # 卷积操作
        conv1_out = self.pool(torch.relu(self.conv1(x)))  # 输出 (batch_size, 32, sequence_length/4)
        conv2_out = self.pool(torch.relu(self.conv2(x)))  # 输出 (batch_size, 32, sequence_length/4)
        conv3_out = self.pool(torch.relu(self.conv3(x)))  # 输出 (batch_size, 32, sequence_length/4)

        # 合并通道
        merged = torch.cat([conv1_out, conv2_out, conv3_out], dim=1)  # 输出 (batch_size, 32*3, sequence_length/4)
        merged = self.bn(merged)  # 批量归一化

        # GRU 输入需要 (batch_size, sequence_length, features)
        merged = merged.permute(0, 2, 1)  # 调整为 (batch_size, sequence_length/4, 32*3)
        gru_out, _ = self.gru(merged)  # 输出 (batch_size, sequence_length/4, 128)

        # 注意力机制
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)  # 输出 (batch_size, sequence_length/4, 128)

        # Flatten
        flatten = attn_out.reshape(attn_out.size(0), -1)  # 使用 reshape 替代 view
        flatten = self.dropout(flatten)

        # 输出层
        outputs = self.fc(flatten)
        return outputs


# Training setup
# 假设 X 和 y 是 PyTorch Tensor，先转换为 NumPy 数组
X_numpy = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
y_numpy = y.cpu().numpy() if isinstance(y, torch.Tensor) else y

# K-fold 分割
k=10
epochs=50
kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)


# criterion = nn.CrossEntropyLoss()
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)  # 预测的概率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

criterion = FocalLoss()
oos_pred = []

from sklearn.metrics import confusion_matrix, classification_report
# 初始化结果列表
# 初始化结果列表
oos_accuracies = []
last_fold_y_true = []
last_fold_y_pred = []
for fold, (train_idx, test_idx) in enumerate(kfold.split(X_numpy, y_numpy), start=1):
    # 直接使用索引选择数据
    train_X, test_X = X_numpy[train_idx], X_numpy[test_idx]
    train_y, test_y = y_numpy[train_idx], y_numpy[test_idx]

    # 创建自定义数据集
    train_dataset = NSL_KDD_Dataset(train_X, train_y)
    test_dataset = NSL_KDD_Dataset(test_X, test_y)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 初始化模型
    model = NSLKDDModel(input_dim=122, num_classes=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        # 使用 tqdm 显示进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_data, batch_labels in progress_bar:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            # 过滤无效标签
            # valid_indices = (batch_labels >= 0) & (batch_labels < 5)
            # batch_data = batch_data[valid_indices]
            # batch_labels = batch_labels[valid_indices]
            #
            # if batch_data.size(0) == 0:  # 跳过空批次
            #     continue

            batch_data = batch_data.unsqueeze(1)  # 添加通道维度
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            # 累积损失和准确性
            epoch_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)

            # 更新进度条
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # 计算每轮的平均损失和准确率
        epoch_loss /= len(train_loader)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

#     # 验证模型
#     model.eval()
#     y_true, y_pred = [], []
#     with torch.no_grad():
#         for batch_data, batch_labels in test_loader:
#             batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
#
#             # 添加通道维度
#             batch_data = batch_data.unsqueeze(1)
#             outputs = model(batch_data)
#             _, preds = torch.max(outputs, 1)
#
#             y_true.extend(batch_labels.cpu().numpy())
#             y_pred.extend(preds.cpu().numpy())
#
#     # 计算验证集的准确率
#     acc = metrics.accuracy_score(y_true, y_pred)
#     oos_pred.append(acc)
#     print(f"Fold Accuracy: {acc}")
#
# # 总体结果
# print(f"Overall Accuracy: {np.mean(oos_pred):.4f}")
#
    # 验证阶段
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            batch_data = batch_data.unsqueeze(1)  # 添加通道维度
            outputs = model(batch_data)
            _, preds = torch.max(outputs, 1)

            y_true.extend(batch_labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # 测试集各类别数量
    # test_class_counts = Counter(y_true)
    # print("\nTest Set Class Distribution:")
    # for label, count in sorted(test_class_counts.items()):
    #     print(f"  Class {label}: {count}")

    # 计算准确率
    acc = metrics.accuracy_score(y_true, y_pred)
    oos_accuracies.append(acc)
    print(f"Fold {fold} Accuracy: {acc:.4f}")

    # 保存最后一折的结果
    if fold == kfold.get_n_splits():
        last_fold_y_true = y_true
        last_fold_y_pred = y_pred

# 保存模型
model_save_path = "../model1.pth"
torch.save(model, model_save_path)
print(f"Complete model saved to {model_save_path}")

# 输出每一折的准确率
print("Fold Accuracies:")
for i, acc in enumerate(oos_accuracies, start=1):
    print(f"  Fold {i}: {acc:.4f}")

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# 定义类别标签
categories = ['DoS', 'Probe', 'U2R', 'R2L', 'Normal']

# 使用最后一折结果计算混淆矩阵
last_cm = confusion_matrix(last_fold_y_true, last_fold_y_pred, labels=range(5))

print("\nLast Fold Confusion Matrix:")
print(last_cm)

# 分类报告
report = classification_report(last_fold_y_true, last_fold_y_pred, target_names=categories)
print("\nClassification Report:")
print(report)

# 初始化总指标和各类指标
detection_rates = {}
false_positive_rates = {}

# 计算总样本数
total_samples = last_cm.sum()

# 初始化用于计算总体指标的 TP、TN、FP、FN
total_TP = 0
total_FP = 0
total_FN = 0
total_TN = 0

# 对每个类别计算 TP, FN, FP, TN
for i, category in enumerate(categories):
    TP = last_cm[i, i]
    FN = last_cm[i, :].sum() - TP
    FP = last_cm[:, i].sum() - TP
    TN = total_samples - (TP + FP + FN)

    # Detection Rate (DR)
    detection_rates[category] = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # False Positive Rate (FPR)
    false_positive_rates[category] = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    # 更新总指标
    total_TP += TP
    total_FP += FP
    total_FN += FN
    total_TN += TN

    # 输出每一类别的检测率和误报率
    print(f"Category: {category}")
    print(f"  Detection Rate (DR): {detection_rates[category]:.4f}")
    print(f"  False Positive Rate (FPR): {false_positive_rates[category]:.4f}")

# 总体指标计算
accuracy = (total_TP + total_TN) / total_samples
overall_dr = total_TP / (total_TP + total_FN)
overall_fpr = total_FP / (total_FP + total_TN)

# 输出总体指标
print(f"\nOverall Accuracy (ACC): {accuracy:.4f}")
print(f"Overall Detection Rate (DR): {overall_dr:.4f}")
print(f"Overall False Positive Rate (FPR): {overall_fpr:.4f}")
