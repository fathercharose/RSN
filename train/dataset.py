import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class MyDataset_patch_normalization(Dataset):
    # Constructor / 构造函数
    def __init__(self, feature_path, label_path):  # __init__ 是一个特殊的方法,在对象被创建时自动调用,用于在创建对象时进行初始化
        super(MyDataset_patch_normalization, self).__init__()   # self 是一个约定俗成的名字，用于表示对象实例本身.使用 self 可以访问该实例的成员变量和方法。
        self.feature_paths = glob.glob(os.path.join(feature_path, '*.npy'))  # Initialize paths to feature and label files / 初始化特征和标签文件的路径
        self.label_paths = glob.glob(os.path.join(label_path, '*.npy'))  # Initialize paths to feature and label files / 初始化特征和标签文件的路径
        self.num_samples = len(self.feature_paths)  # Record the size of the dataset / 记录数据集的大小

    # Return the size of the dataset / 返回数据集大小
    def __len__(self):
        return self.num_samples


    # Return data and label for the given index / 返回给定索引的数据和标签
    def __getitem__(self, index):   # 当使用 obj[index] 时，会调用对象的 __getitem__ 方法，用于获取对象的元素，使对象可以通过索引或键访问。
        feature_data = np.load(self.feature_paths[index])
        feature_data_abs = np.abs(feature_data)                    # Normalize to [-1, 1] / 归负一到一化
        feature_data = feature_data / np.max(feature_data_abs)    # Normalize to [-1, 1] / 归负一到一化
        label_data = np.load(self.label_paths[index])
        label_data_abs = np.abs(label_data)                       # Normalize to [-1, 1] / 归负一到一化
        label_data = label_data / np.max(label_data_abs)           # Normalize to [-1, 1] / 归负一到一化
        feature_data = torch.from_numpy(feature_data)  # Convert to tensor / numpy转成张量
        label_data = torch.from_numpy(label_data)
        feature_data.unsqueeze_(0)  # Add a dimension 128*128 => 1*128*128 / 增加一个维度128*128 =>1*128*128
        label_data.unsqueeze_(0)   # Add a dimension 1*128*128 => 1*1*128*128 / 增加一个维度1*128*128 =>1*1*128*128
        return feature_data, label_data



