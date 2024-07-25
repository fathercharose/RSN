import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

noised_data = np.load("../data/compare_section/sampley_noised_denoised.npy")
Ceemd_data = np.load("../data/compare_section/Ceemd_sampley.npy")
Unet_data = np.load("../data/compare_section/F3_Unet_100_LR002_sample_y.npy")
Drsn_data = np.load("../data/compare_section/F3_100_LR002_sample_y.npy")


vmin1 = -8000    # Minimum value
vmax1 = 8000   # Maximum value


fontsize1 = 12
fontsize_ticks = 10  # 刻度字体大小

fig = plt.figure(figsize=(12, 12))  # 调整整体图的大小
gs = GridSpec(2, 3, width_ratios=[1, 1, 0.05], wspace=0.05)  # 调整网格布局

# 第一行，左边图像
ax0 = plt.subplot(gs[0, 0])
im0 = ax0.imshow(Ceemd_data, cmap=plt.cm.seismic, aspect='auto', vmin=vmin1, vmax=vmax1)
ax0.set_title("Noised", fontsize=fontsize1)
ax0.set_xlabel("Traces", fontsize=fontsize1)
ax0.set_ylabel("Samples", fontsize=fontsize1)
ax0.text(0.05, 1.05, "(a)", transform=ax0.transAxes, fontsize=fontsize1, va='top', ha='right')
plt.setp(ax0.get_xticklabels(), fontsize=fontsize_ticks)
plt.setp(ax0.get_yticklabels(), fontsize=fontsize_ticks)


# 第一行，右边图像
ax1 = plt.subplot(gs[0, 1], sharey=ax0)  # 共享 y 轴
im1 = ax1.imshow(Drsn_data, cmap=plt.cm.seismic, aspect='auto', vmin=vmin1, vmax=vmax1)
ax1.set_title("CEEMD method", fontsize=fontsize1)
ax1.set_xlabel("Traces", fontsize=fontsize1)
ax1.text(0.05, 1.05, "(b)", transform=ax1.transAxes, fontsize=fontsize1, va='top', ha='right')
ax1.yaxis.set_visible(False)  # 去掉 y 轴坐标
plt.setp(ax1.get_xticklabels(), fontsize=fontsize_ticks)


# 第二行，左边图像
ax2 = plt.subplot(gs[1, 0])
im2= ax2.imshow(Unet_data, cmap=plt.cm.seismic, aspect='auto', vmin=vmin1, vmax=vmax1)
ax2.set_title("U-net method", fontsize=fontsize1)
ax2.set_xlabel("Traces", fontsize=fontsize1)
ax2.set_ylabel("Samples", fontsize=fontsize1)
ax2.text(0.05, 1.05, "(c)", transform=ax2.transAxes, fontsize=fontsize1, va='top', ha='right')
plt.setp(ax2.get_xticklabels(), fontsize=fontsize_ticks)
plt.setp(ax2.get_yticklabels(), fontsize=fontsize_ticks)

# 第二行 右边图像
ax3 = plt.subplot(gs[1, 1], sharey=ax0)  # 共享 y 轴
im3 = ax3.imshow(Drsn_data, cmap=plt.cm.seismic, aspect='auto', vmin=vmin1, vmax=vmax1)
ax3.set_title("Proposed method", fontsize=fontsize1)
ax3.set_xlabel("Traces", fontsize=fontsize1)
ax3.text(0.05, 1.05, "(d)", transform=ax3.transAxes, fontsize=fontsize1, va='top', ha='right')
ax3.yaxis.set_visible(False)  # 去掉 y 轴坐标
plt.setp(ax3.get_xticklabels(), fontsize=fontsize_ticks)

# Colorbar
cax2 = plt.subplot(gs[:, 2])
cbar = fig.colorbar(im1, cax=cax2, extend='both')
cbar.ax.tick_params(labelsize=fontsize_ticks)

plt.subplots_adjust(wspace=0.1, hspace=0.2)




plt.show()