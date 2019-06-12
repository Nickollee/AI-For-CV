# ！/usr/bin/env python
# -*- coding:utf-8 -*-
# author:Niki
"""
week6 assinment:a tiny project:cars classification
2019.6.1
win10
GTX-1060
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, datasets, models
import visdom
import time
import os
import numpy as np
import torchvision
import matplotlib.pyplot as plt

viz = visdom.Visdom()

BATCH_SIZE = 4
LR = 0.001
EPOCHS = 30

USE_GPU = True
if USE_GPU:
    gpu_status = torch.cuda.is_available()
else:
    gpu_status = False


data_path = "E:\***\a_tiny_project\\"
data_transforms = {
    "train":transforms.Compose([
        transforms.Resize(224),#缩放图片，长宽比保持，最短边长224像素
        transforms.CenterCrop(224),#从中间切出224*224的图片
        transforms.ToTensor(),#图片转换为tensor，归一化至[0,1]
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])#标准化normalize   channel=（channel-mean）/std至[-1,1]
    ]),
    "test":transforms.Compose([
        transforms.Resize(256),# 图片大小缩放 统一图片格式
        transforms.CenterCrop(224),# 从中心裁剪
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
}
#图像变换
image_datasets = {x:datasets.ImageFolder(os.path.join(data_path,x),data_transforms[x])for x in ["train","test"]}
#导入图像
data_loaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True) for x in ['train', 'test']}
data_sizes = {x:len(image_datasets[x])for x in ['train', 'test']}#训练集和测试集数量
class_names = image_datasets['train'].classes#类别名（子文件夹名）
print(data_sizes, class_names)
inputs, classes = next(iter(data_loaders['test']))#取一个batch的样本操作
#可视化部分图片
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(1)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(data_loaders['train']))   # 取一个batch的样本操作

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)   # 此时的输入为Tensor

imshow(out, title=[class_names[x] for x in classes])

class CNN(nn.Module):
    def __init__(self, in_dim, n_class):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(True),
            nn.Conv2d(in_dim, 16, 7), # 224 >> 218
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 218 >> 109
            nn.ReLU(True),
            nn.Conv2d(16, 32, 5),  # 105
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5),  # 101
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 101 >> 50
            nn.Conv2d(64, 128, 3, 1, 1),  #
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(3),  # 50 >> 16
        )
        self.fc = nn.Sequential(
            nn.Linear(128*16*16, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(True),
            nn.Linear(120, n_class))
    def forward(self, x):
        out = self.cnn(x)
        out = self.fc(out.view(-1, 128*16*16))
        return out

# 输入3层rgb ，输出 分类 3
model = CNN(3, 3)
model = torchvision.models.resnet18(pretrained=True)#调用包含训练好的参数的模型
# for param in model.parameters():
#     param.requires_grad = False
num_ftrs = model.fc.in_features#最后一个全连接的输入维度
model.fc = nn.Linear(num_ftrs, 3)#将最后一个全连接由（512,1000）修改为（512,3）因为原网络分为1000类自己的数据集分为3个类别


if gpu_status:
    net = model.cuda()#网络变量调用GPU
    print("使用gpu")
else:
    print("使用cpu")
# 可视化
line = viz.line(Y=np.arange(10))
loss_f = nn.CrossEntropyLoss()#交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)#优化器SGD，单参数组
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)#每7个epoch衰减0.1倍

start_time = time.time()
best_model_wts = model.state_dict()#加载参数
best_acc = 0.0
train_loss, test_loss, train_acc, test_acc, time_p = [], [], [], [], []
for epoch in range(EPOCHS):
    # Each epoch has a training and test phase
    for phase in ['train', 'test']:
        if phase == 'train':
            scheduler.step()#训练的时候进行学习率规划，定义给出
            model.train(True)#设置为训练的模式
        else:
            model.train(False)#设置为测试模式
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for data in data_loaders[phase]:
            inputs, labels = data#get the input

            if gpu_status:# wrap them in Variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()#参数梯度置零
            outputs = model(inputs)#前向传播
            preds = torch.max(outputs.data, 1)[1]
            loss = loss_f(outputs, labels)
            if phase == 'train':
                loss.backward()#如果是训练过程，反向传播和优化
                optimizer.step()

            running_loss += loss.item()*len(labels)#把loss.data[0]改为了loss.item()
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / data_sizes[phase]
        epoch_acc = running_corrects.float() / data_sizes[phase]#！！！

        if phase == 'test':
            test_loss.append(epoch_loss)
            test_acc.append(epoch_acc)
        else:
            train_loss.append(epoch_loss)
            train_acc.append(epoch_acc)

        if phase == 'test' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model.state_dict()

    print("[{}/{}] train_loss:{:.5f}|train_acc:{:.5f}|test_loss:{:.5f}|test_acc{:.5f}".format(epoch+1, EPOCHS,
                                               train_loss[-1], train_acc[-1], test_loss[-1], test_acc[-1]))
    time_p.append(time.time()-start_time)

time_elapsed = time.time() - start_time
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Best test Acc: {:4f}'.format(best_acc))

# load best model weights
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), "E:\***\1.py")
