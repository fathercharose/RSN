import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

noised_data = np.load("..\\data\\test\\sampley17_128_300_59noised.npy")
raw_data = np.load("..\\data\\test\\labely17_128_300_59.npy")
# 加载原始数据和加噪数据
vmin1 = -8000    # Minimum value
vmax1 = 8000   # Maximum value
# 设置设备为GPU或CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# #### 加载模型1
model1 = torch.load("F3_snr4_14_without_StepLR\\weight_100\\model_epoch30_temp.pth")   # 选用模型
model1.to(device=device)  # 模型拷贝至GPU # Move the model to GPU
model1.eval()  # 开启评估模式  # Enable evaluation mode
# 数据归一化
noised_data_abs = noised_data
noised_data_abs = np.abs(noised_data_abs)
noised_data1 = noised_data / np.max(noised_data_abs)   # Normalization
patch1 = torch.from_numpy(noised_data1)  # python转换为tensor  # Convert numpy to tensor
patch1 = patch1.unsqueeze_(0)
patch1 = patch1.unsqueeze_(0)
print((patch1.shape))
# patch = patch.reshape(1, patch.shape[0], patch.shape[1])  # 对数据维度进行扩充(批量，通道，高，宽)

patch1 = patch1.to(device=device, dtype=torch.float32)  # 数据拷贝至GPU
predict_data1 = model1(patch1)  # 预测结果  # Prediction result
predict_data1 = predict_data1.squeeze_(0)
predict_data1 = predict_data1.squeeze_(0)

predict_data1 = predict_data1.detach().cpu().numpy()  # 将数据从GPU中拷贝出来，放入CPU中，并转换为numpy数组  # Copy data from GPU to CPU and convert to numpy array
predict_data1 = predict_data1 * np.max(noised_data_abs)  # Denormalize

# ##### 加载模型2
model2 = torch.load("F3_snr4_14_without_StepLR_Maxmin\\weight_100\\model_epoch30_temp.pth")   # 选用模型
model2.to(device=device)  # 模型拷贝至GPU # Move the model to GPU
model2.eval()  # 开启评估模式  # Enable evaluation mode
# 数据归一化
feature_data = (noised_data - np.min(noised_data)) / (np.max(noised_data) - np.min(noised_data))   # Normalization
patch2 = torch.from_numpy(feature_data)  # python转换为tensor  # Convert numpy to tensor
patch2 = patch2.unsqueeze_(0)
patch2 = patch2.unsqueeze_(0)
print((patch2.shape))
# patch = patch.reshape(1, patch.shape[0], patch.shape[1])  # 对数据维度进行扩充(批量，通道，高，宽)

patch2 = patch2.to(device=device, dtype=torch.float32)  # 数据拷贝至GPU
predict_data2 = model2(patch2)  # 预测结果  # Prediction result
predict_data2 = predict_data2.squeeze_(0)
predict_data2 = predict_data2.squeeze_(0)

predict_data2 = predict_data2.detach().cpu().numpy()  # 将数据从GPU中拷贝出来，放入CPU中，并转换为numpy数组  # Copy data from GPU to CPU and convert to numpy array
predict_data2 = predict_data2 * (np.max(noised_data) - np.min(noised_data)) + np.min(noised_data)  # Denormalize


############ 两个振幅曲线同一个图中展示
fontsize1 = 10
fontsize_ticks = 8  # 刻度字体大小

fig = plt.figure(figsize=(12, 6))  # 调整整体图的大小
gs = GridSpec(2, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.05)  # 调整网格布局

# 第一行，左边图像
ax0 = plt.subplot(gs[0, 0])
im1 = ax0.imshow(raw_data, cmap=plt.cm.seismic, aspect='auto', vmin=vmin1, vmax=vmax1)
ax0.set_title("Original", fontsize=fontsize1)
ax0.set_xlabel("Traces", fontsize=fontsize1)
ax0.set_ylabel("Samples", fontsize=fontsize1)
ax0.text(0.03, 1.1, "(a)", transform=ax0.transAxes, fontsize=fontsize1, va='top', ha='right')
plt.setp(ax0.get_xticklabels(), fontsize=fontsize_ticks)
plt.setp(ax0.get_yticklabels(), fontsize=fontsize_ticks)

# 第一行，中间图像
ax1 = plt.subplot(gs[0, 1], sharey=ax0)  # 共享 y 轴
im2 = ax1.imshow(predict_data1, cmap=plt.cm.seismic, aspect='auto', vmin=vmin1, vmax=vmax1)
ax1.set_title("Absolute maximum normalization", fontsize=fontsize1)
ax1.set_xlabel("Traces", fontsize=fontsize1)
ax1.text(0.03, 1.1, "(b)", transform=ax1.transAxes, fontsize=fontsize1, va='top', ha='right')
ax1.yaxis.set_visible(False)  # 去掉 y 轴坐标
plt.setp(ax1.get_xticklabels(), fontsize=fontsize_ticks)

# 第一行，右边图像
ax2 = plt.subplot(gs[0, 2], sharey=ax0)  # 共享 y 轴
im3 = ax2.imshow(predict_data2, cmap=plt.cm.seismic, aspect='auto', vmin=vmin1, vmax=vmax1)
ax2.set_title("Min-max normalization", fontsize=fontsize1)
ax2.set_xlabel("Traces", fontsize=fontsize1)
ax2.text(0.03, 1.1, "(c)", transform=ax2.transAxes, fontsize=fontsize1, va='top', ha='right')
ax2.yaxis.set_visible(False)  # 去掉 y 轴坐标
plt.setp(ax2.get_xticklabels(), fontsize=fontsize_ticks)

# Colorbar
cax2 = plt.subplot(gs[0, 3])
cbar = fig.colorbar(im3, cax=cax2, extend='both')
cbar.ax.tick_params(labelsize=fontsize_ticks)
cbar.set_ticks([-8000, -6000, -4000, -2000, 0, 2000, 4000, 6000, 8000])

# 第二行，将两个曲线放在一起
ax_combined = plt.subplot(gs[1, :])  # 创建一个横跨第二行所有列的子图
ax_combined.plot(raw_data[:, 220], label="Original", color="black")
ax_combined.plot(predict_data1[:, 220], label="Absolute maximum normalization", color="blue")
ax_combined.plot(predict_data2[:, 220], label="Min-max normalization", color="red")
ax_combined.set_title("Trace 220", fontsize=fontsize1)
ax_combined.set_xlabel("Samples", fontsize=fontsize1)
ax_combined.set_ylabel("Amplitude", fontsize=fontsize1)
ax_combined.set_yticks([-8000, -6000, -4000, -2000, 0, 2000, 4000, 6000, 8000])
# ax_combined.legend(fontsize=fontsize_ticks)
ax_combined.text(0.01, 1.1, "(d)", transform=ax_combined.transAxes, fontsize=12, va='top', ha='right')
plt.setp(ax_combined.get_xticklabels(), fontsize=fontsize_ticks)
plt.setp(ax_combined.get_yticklabels(), fontsize=fontsize_ticks)
plt.legend(fontsize='6', bbox_to_anchor=(0.8, 1), loc='upper left')

# 进一步细调子图间距
plt.subplots_adjust(wspace=0.6, hspace=0.4)

plt.show()