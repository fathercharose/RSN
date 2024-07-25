from model.unet import UNet
from dataset import MyDataset_patch_normalization
from torch import optim
import torch.nn as nn
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm


#  # 超参数设置
save_path = "weight\\"   # 保存训练后模型的路径  # Path to save trained models
directory1 = os.path.dirname(save_path)
if not os.path.exists(directory1):
    os.makedirs(directory1)
# 指定特征和标签数据地址，加载数据集
train_path_x = "..\\data\\train\\samples\\"  #训练集样本路径  # Training set sample path
train_path_y = "..\\data\\train\\labels\\"           # 训练集标签路径 # Training set label path
train_samples = 20
valida_size = 100
batch_size = 5
# 定义优化方法
epochs = 30  # 设置训练次数  # Number of training cycles
LR = 0.02   # 设置学习率# Learning rate
# 选择设备，有cuda用cuda，没有就用cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载网络
my_net = UNet(1)
# 将网络拷贝到设备中
my_net.to(device=device)
optimizer = optim.Adam(my_net.parameters(), lr=LR)
# 定义损失函数
criterion = nn.MSELoss(reduction='sum')  # reduction='sum'表示不除以batch_size
# 定义学习率调度器，每5个epoch学习率减半
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# 划分数据集，训练集：验证集 = 8:1
full_dataset = MyDataset_patch_normalization(train_path_x, train_path_y)
train_size = int(len(full_dataset) - valida_size)
print("train_size=", train_size)


# 划分数据集
train_dataset, valida_dataset = torch.utils.data.random_split(full_dataset,
                                                                         [train_size, valida_size])


#####
def collate_fn(batch):
    # 获取batch中所有feature和label的尺寸
    feature_sizes = [item[0].shape for item in batch]
    label_sizes = [item[1].shape for item in batch]

    # 找到最大的feature和label尺寸
    max_feature_size = tuple(max(s[i] for s in feature_sizes) for i in range(len(feature_sizes[0])))
    max_label_size = tuple(max(s[i] for s in label_sizes) for i in range(len(label_sizes[0])))

    # 定义一个空的tensor来存储所有feature和label
    features = torch.zeros((len(batch),) + max_feature_size)
    labels = torch.zeros((len(batch),) + max_label_size)

    # 将每个feature和label填充到对应的tensor中
    for i, item in enumerate(batch):
        feature = item[0]
        label = item[1]

        # 计算填充的数量
        feature_pad = (0, max_feature_size[-1] - feature.shape[-1])
        label_pad = (0, max_label_size[-1] - label.shape[-1])

        # 填充feature和label
        features[i] = torch.nn.functional.pad(feature, feature_pad)
        labels[i] = torch.nn.functional.pad(label, label_pad)

    return features, labels
#####
# 加载并且乱序训练数据集
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
# 加载并且乱序验证数据集
valida_loader = torch.utils.data.DataLoader(dataset=valida_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)



temp_sets1 = []  # 用于记录训练，验证集的loss,每一个epoch都做一次训练，验证 # Used to record the loss of training and validation sets, training and validation are performed for each epoch


start_time_str = time.strftime("1. %Y-%m-%d %H:%M:%S", time.localtime())  # 开始时间  # Start time
start_time = time.time()  # 记录开始时间戳

# 每一个epoch都做一次训练，验证，测试 # Perform training, validation, and testing for each epoch
for epoch in range(epochs):
    # 训练集训练网络   # Train the network on the training set
    train_loss = 0.0
    my_net.train()  # 开启训练模式  # Enable training mode
    train_loader_iter = tqdm(enumerate(train_loader, 0), total=min(len(train_loader), train_samples),
                             desc=f'Epoch {epoch + 1}/{epochs}', unit='batch')
    for batch_idx1, (batch_x, batch_y) in train_loader_iter:  # 0开始计数   # Start counting from 0
        if batch_idx1 >= train_samples:  # 只运行前1000个数据
            break
        # 加载数据至GPU
        batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.float32)
        err_out1 = my_net(batch_x)  # 使用网络参数，输出预测结果  # Use network parameters to output predictions
        # 计算loss
        loss1 = criterion(err_out1, batch_y)
        train_loss += loss1.item()  # 累加计算本次epoch的loss，最后还需要除以每个epoch可以抽取多少个batch数，即最后的n_count值  # Accumulate the loss of this epoch, and finally need to divide by the number of batches that can be extracted per epoch, that is, the final value of n_count
        optimizer.zero_grad()  # 先将梯度归零,等价于net.zero_grad()   # Reset gradients to zero, equivalent to net.zero_grad(0
        loss1.backward()  # 反向传播计算得到每个参数的梯度值  # Backpropagation to calculate gradients for each parameter
        optimizer.step()  # 通过梯度下降执行一步参数更新 # Execute one step parameter update through gradient descent
    train_loss = train_loss / batch_idx1 # 本次epoch的平均loss # Average loss of this epoch

    # 验证集验证网络
    my_net.eval()  # 开启评估/测试模式
    val_loss = 0.0

    valida_loader_iter = tqdm(enumerate(valida_loader, 0), total=len(valida_loader),
                              desc=f'Validation {epoch + 1}/{epochs}', unit='batch')
    for batch_idx2, (val_x, val_y) in valida_loader_iter:
        # 加载数据至GPU
        val_x = val_x.to(device=device, dtype=torch.float32)
        val_y = val_y.to(device=device, dtype=torch.float32)
        with torch.no_grad():  # 不需要做梯度更新，所以要关闭求梯度
            err_out2 = my_net(val_x)  # 使用网络参数，输出预测结果
            # 计算loss
            loss2 = criterion(err_out2, val_y)
            val_loss += loss2.item()  # 累加计算本次epoch的loss，最后还需要除以每个epoch可以抽取多少个batch数，即最后的count值
    val_loss = val_loss / (batch_idx2 + 1)

    # 训练，验证，测试的loss保存至loss_sets中     # Save the loss of training, validation, and testing to loss_sets
    loss_set = [train_loss, val_loss]
    temp_sets1.append(loss_set)
    # {:.4f}值用format格式化输出，保留小数点后四位
    print("epoch={}，训练集loss：{:.4f}，验证集loss：{:.4f}".format(epoch + 1, train_loss, val_loss))

    # 保存网络模型  # Save the network model
    model_name = f'model_epoch{epoch + 1}'  # 模型命名
    os.makedirs(f'{save_path}models_epoches', exist_ok=True)  # torch.save 函数不会自动创建保存路径
    torch.save(my_net, os.path.join(f'{save_path}models_epoches', model_name + '_temp.pth'))  # 保存整个神经网络的模型结构以及参数

end_time_str = time.strftime("1. %Y-%m-%d %H:%M:%S", time.localtime())  # 结束时间
end_time = time.time()  # 记录结束时间戳

# 计算训练时长
elapsed_time = end_time - start_time
elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

# 将训练花费的时间写成一个txt文档，保存到当前文件夹下
with open(f'{save_path}训练时间.txt', 'w', encoding='utf-8') as f:
    f.write(f"开始时间: {start_time_str}\n")
    f.write(f"结束时间: {end_time_str}\n")
    f.write(f"训练时长: {elapsed_time_str}\n")
print("训练开始时间{}>>>>>>>>>>>>>>>>训练结束时间{}，训练时长{}".format(start_time_str, end_time_str,
                                                                       elapsed_time_str))  # 打印所用时间

# temp_sets1是三维张量无法保存，需要变成2维数组才能存为txt文件
loss_sets = []
for sets in temp_sets1:
    for i in range(2):
        loss_sets.append(sets[i])
loss_sets = np.array(loss_sets).reshape(-1, 2)  # 重塑形状10*2，-1表示自动推导
# fmt参数，指定保存的文件格式。将loss_sets存为txt文件
np.savetxt(f'{save_path}loss_sets.txt', loss_sets, fmt='%.4f')


# 显示loss曲线
loss_lines = np.loadtxt(f'{save_path}loss_sets.txt')
# 前面除以batch_size会导致数值太小了不易观察
train_line = loss_lines[:, 0] / batch_size
valida_line = loss_lines[:, 1] / batch_size
x1 = range(len(train_line))
fig1 = plt.figure()
plt.plot(x1, train_line, x1, valida_line)
# plt.title('batch_size={}'.format(batch_size))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'valida'])
plt.savefig(f'{save_path}loss_plot.png', bbox_inches='tight')
plt.tight_layout()


plt.show()
exit()