import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset



import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class MyDataset_patch_normalization_Maxmin(Dataset):
    # Constructor / 构造函数
    def __init__(self, feature_path, label_path):
        super(MyDataset_patch_normalization_Maxmin, self).__init__()
        self.feature_paths = glob.glob(os.path.join(feature_path, '*.npy'))  # Initialize paths to feature and label files / 初始化特征和标签文件的路径
        self.label_paths = glob.glob(os.path.join(label_path, '*.npy'))  # Initialize paths to feature and label files / 初始化特征和标签文件的路径
        self.num_samples = len(self.feature_paths)  # Record the size of the dataset / 记录数据集的大小

    # Return the size of the dataset / 返回数据集大小
    def __len__(self):
        return self.num_samples

    # Return data and label for the given index / 返回给定索引的数据和标签
    def __getitem__(self, index):
        feature_data = np.load(self.feature_paths[index])
        label_data = np.load(self.label_paths[index])

        # Normalize to [0, 1] / 归一到0到1
        feature_data = (feature_data - np.min(feature_data)) / (np.max(feature_data) - np.min(feature_data))
        label_data = (label_data - np.min(label_data)) / (np.max(label_data) - np.min(label_data))

        feature_data = torch.from_numpy(feature_data)  # Convert to tensor / numpy转成张量
        label_data = torch.from_numpy(label_data)

        feature_data.unsqueeze_(0)  # Add a dimension 128*128 => 1*128*128 / 增加一个维度128*128 =>1*128*128
        label_data.unsqueeze_(0)   # Add a dimension 1*128*128 => 1*1*128*128 / 增加一个维度1*128*128 =>1*1*128*128

        return feature_data, label_data
