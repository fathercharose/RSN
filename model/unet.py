import torch
from torch import nn
from torch.nn import functional as F

# 定义一个卷积块类，包括两次卷积操作和激活函数
class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            # 第一次卷积，3x3卷积核，步幅为1，填充方式为'reflect'，不使用偏置
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),  # 批量归一化
            nn.LeakyReLU(),  # 激活函数
            # 第二次卷积，3x3卷积核，步幅为1，填充方式为'reflect'，不使用偏置
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),  # 批量归一化
            nn.LeakyReLU()  # 激活函数
        )

    # 前向传播函数
    def forward(self, x):
        return self.layer(x)


# 定义一个下采样类，包括一次卷积操作和激活函数
class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            # 卷积操作，3x3卷积核，步幅为2，进行下采样，填充方式为'reflect'，不使用偏置
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),  # 批量归一化
            nn.LeakyReLU()  # 激活函数
        )

    # 前向传播函数
    def forward(self, x):
        return self.layer(x)


# 定义一个上采样类，包括一次卷积操作
class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        # 卷积操作，1x1卷积核，步幅为1
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)

    # 前向传播函数，包括上采样和拼接操作
    def forward(self, x, feature_map):
        # 使用最近邻插值方法进行上采样，放大2倍
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        # 卷积操作
        out = self.layer(up)
        # 将上采样后的特征图与对应的下采样阶段的特征图拼接
        return torch.cat((out, feature_map), dim=1)


# 定义一个UNet类，包括网络结构的定义和前向传播函数
class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        # 定义编码器部分的卷积块和下采样操作
        self.c1 = Conv_Block(1, 64)
        self.d1 = DownSample(64)
        self.c2 = Conv_Block(64, 128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128, 256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256, 512)
        self.d4 = DownSample(512)
        self.c5 = Conv_Block(512, 1024)
        # 定义解码器部分的上采样操作和卷积块
        self.u1 = UpSample(1024)
        self.c6 = Conv_Block(1024, 512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.u4 = UpSample(128)
        self.c9 = Conv_Block(128, 64)
        # 定义输出层的卷积操作
        self.out = nn.Conv2d(64, num_classes, 3, 1, 1)

    # 前向传播函数，包括编码和解码过程
    def forward(self, x):
        # 编码过程，获取各个层的特征图
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        # 解码过程，逐步上采样并拼接特征图
        O1 = self.c6(self.u1(R5, R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))
        # 输出结果
        return self.out(O4)


# 测试网络结构
if __name__ == '__main__':
    x = torch.randn(1, 1, 256, 256)  # 创建一个随机输入张量
    net = UNet(1)  # 创建UNet网络实例
    print(net(x).shape)  # 打印网络输出的形状
