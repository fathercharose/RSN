import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# SoftThreshold模块用于计算输入张量的软阈值
# SoftThreshold module for computing the soft threshold of the input tensor
class SoftThreshold(nn.Module):
    def __init__(self, channels):
        super(SoftThreshold, self).__init__()
        # 使用自适应平均池化层对输入进行池化操作
        # Using adaptive average pooling layer to pool the input
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全连接层实现对输入通道的线性变换
        # Fully connected layer to perform linear transformation on input channels
        self.fc = nn.Sequential(
            nn.Linear(channels, channels, bias=True),  # 输入输出大小相同的全连接层
            nn.BatchNorm1d(channels),  # 批标准化层
            nn.ReLU(inplace=True),  # 非线性激活函数ReLU
            nn.Linear(channels, channels, bias=True),  # 输入输出大小相同的全连接层
            nn.Sigmoid()  # Sigmoid激活函数
        )

    def forward(self, x):
        abs_x = torch.abs(x)  # 计算输入张量的绝对值
        avg_x = self.avg_pool(abs_x).view(x.size(0), -1)  # 平均池化操作后将张量形状调整为(batch_size, num_channels)
        alpha = self.fc(avg_x)  # 经过全连接层计算软阈值的系数alpha
        threshold = alpha * avg_x  # 计算软阈值
        threshold = threshold.view(x.size(0), x.size(1), 1, 1)  # 将软阈值扩展为与输入张量相同的形状
        return torch.sign(x) * torch.relu(torch.abs(x) - threshold)  # 返回软阈值处理后的张量


# 残差收缩模块
# Shrinkage Mapping module
class ShrinkageMapping(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ShrinkageMapping, self).__init__()
        # 第一个卷积层
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        # 批标准化层
        # Batch normalization layer
        self.BN1 = nn.BatchNorm2d(out_channels)
        # 非线性激活函数ReLU
        # Non-linear activation function ReLU
        self.Relu1 = nn.ReLU(inplace=True)
        # 第二个卷积层
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        # 批标准化层
        # Batch normalization layer
        self.BN2 = nn.BatchNorm2d(out_channels)
        # 非线性激活函数ReLU
        # Non-linear activation function ReLU
        self.Relu2 = nn.ReLU(inplace=True)
        # SoftThreshold模块
        # SoftThreshold module
        self.soft_threshold = SoftThreshold(out_channels)
        # 非线性激活函数ReLU
        # Non-linear activation function ReLU
        self.Relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x  # 捷径连接
        out = self.BN1(self.conv1(x))  # 第一个卷积层操作
        out = self.Relu1(out)  # ReLU激活函数
        out = self.conv2(out)  # 第二个卷积层操作
        out = self.BN2(out)  # 批标准化层
        out = self.Relu2(out)  # ReLU激活函数
        out = self.soft_threshold(out)  # 软阈值处理
        out = identity + out  # 捷径连接加和
        out = self.Relu3(out)  # ReLU激活函数
        return out


# Deep Residual Shrinkage Network
# 深度残差收缩网络
class DRSN_CS(nn.Module):
    def __init__(self, upscale_factor=2):
        super(DRSN_CS, self).__init__()
        # 网络的头部部分，用于提取低级特征
        # Head of the network for extracting low-level features
        self.head = nn.Sequential(
            # 第一个卷积层，输入通道数为1，输出通道数为64，卷积核大小为3x3，填充为1
            # First convolutional layer, input channels: 1, output channels: 64, kernel size: 3x3, padding: 1
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(8),  # 批标准化层
            nn.ReLU(inplace=True),  # 非线性激活函数ReLU
            # 第二个卷积层，输入通道数为64，输出通道数为64，卷积核大小为3x3，填充为1
            # Second convolutional layer, input channels: 64, output channels: 64, kernel size: 3x3, padding: 1
            nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),  # 批标准化层
            nn.ReLU(inplace=True),  # 非线性激活函数ReLU
        )

        # 网络的身体部分，用于提取高级特征
        # Body of the network for extracting high-level features
        self.body = nn.Sequential(
            # 第一个卷积层，输入通道数为64，输出通道数为128，卷积核大小为3x3，步长为2，填充为1
            # First convolutional layer, input channels: 64, output channels: 128, kernel size: 3x3, stride: 2, padding: 1
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),  # 批标准化层
            nn.ReLU(inplace=True),  # 非线性激活函数ReLU

            # 第二个卷积层，输入通道数为128，输出通道数为256，卷积核大小为3x3，步长为2，填充为1
            # Second convolutional layer, input channels: 128, output channels: 256, kernel size: 3x3, stride: 2, padding: 1
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),  # 批标准化层
            nn.ReLU(inplace=True),  # 非线性激活函数ReLU

        )

        self.upscale = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),  # 批标准化层
            nn.ReLU(inplace=True),  # 非线性激活函数ReLU

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),  # 批标准化层
            nn.ReLU(inplace=True),  # 非线性激活函数ReLU

            nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(8),  # 批标准化层
            nn.ReLU(inplace=True),  # 非线性激活函数ReLU
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True),

        )


    def forward(self, x1):
        x2 = self.head(x1)
        x3 = self.body(x2)
        x4 = self.upscale(x3)
        return x4  # 返回输出张量


# 用于测试，如果作为模块导入，则执行以下代码，否则不执行
if __name__ == '__main__':
    x = torch.randn(5, 1, 256, 256, requires_grad=True)  # 输入张量
    model = DRSN_CS()  # 创建DRSN_CS模型实例
    y = model(x)  # 模型前向传播
    print(y.shape)  # 输出张量形状
    exit()

