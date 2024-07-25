import numpy as np
import torch
import math
import os
import matplotlib.pyplot as plt

# 定义PSNR计算函数
def psnr(signal, noise_data):
    signal = np.array(signal)
    noise_data = np.array(noise_data)
    mse = np.mean((signal - noise_data) ** 2)
    if mse == 0:
        return float('inf')  # 防止除以0的情况
    return 20 * math.log10(np.max(signal) / np.sqrt(mse))

# 获取数据文件夹路径
Ceemd_folder = "..\\data\\snr_compare\\CEEMD_snr2_9\\"
Unet_folder = "..\\data\\snr_compare\\F3_Unet_StepLR\\weight_200_LR002\\"
DRSN_folder = "..\\data\\snr_compare\\F3_StepLR_1\\weight_200_LR002\\"

# 初始化PSNR列表
psnr_Ceemd = []
psnr_Unet = []
psnr_DRSN = []

# 获取所有文件名
label_data = np.load("..\\data\\snr_compare\\Original_labely171_36_326_60.npy")
Ceemd_files = sorted(os.listdir(Ceemd_folder))
Unet_files = sorted(os.listdir(Unet_folder))
DRSN_files = sorted(os.listdir(DRSN_folder))

# 确保三个文件夹中的文件数量相同
assert len(Ceemd_files) == len(Unet_files) == len(DRSN_files), "文件夹中的文件数量不匹配"

# 遍历所有文件
for filename1, filename2, filename3 in zip(Ceemd_files, Unet_files, DRSN_files):
    if filename1.endswith('.npy') and filename2.endswith('.npy') and filename3.endswith('.npy'):
        # 加载原始数据、加噪数据和预测数据
        Ceemd_file_path = os.path.join(Ceemd_folder, filename1)
        Unet_file_path = os.path.join(Unet_folder, filename2)
        DRSN_file_path = os.path.join(DRSN_folder, filename3)

        Ceemd__data = np.load(Ceemd_file_path)
        Unet_data = np.load(Unet_file_path)
        DRSN_data = np.load(DRSN_file_path)

        # 计算预测前后的PSNR
        psnr_Ceemd_value = psnr(label_data, Ceemd__data)
        psnr_Unet_value = psnr(label_data, Unet_data)
        psnr_DRSN_value = psnr(label_data, DRSN_data)

        # 存储PSNR值
        psnr_Ceemd.append(psnr_Ceemd_value)
        psnr_Unet.append(psnr_Unet_value)
        psnr_DRSN.append(psnr_DRSN_value)



# 绘制PSNR比较图
fontsize2 = 14
plt.figure(figsize=(10, 6))
plt.plot(range(2, len(psnr_Ceemd) + 2), psnr_Ceemd, label='CEEMD method', marker='o')
plt.plot(range(2, len(psnr_Unet) + 2), psnr_Unet, label='U-net method', marker='s')
plt.plot(range(2, len(psnr_DRSN) + 2), psnr_DRSN, label='Proposed method', marker='^')
plt.xlabel('SNR (dB)', fontsize=fontsize2)
plt.ylabel('PSNR (dB)', fontsize=fontsize2)
plt.title('PSNR vs. SNR', fontsize=fontsize2)
plt.legend()
plt.tight_layout()

plt.show()
