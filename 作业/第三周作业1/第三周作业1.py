# -*- coding: utf-8 -*-
'''
第三周作业1：
1.  采用步进(Step into)的调试方法从创建网络模型开始（net = LeNet(classes=2)）进入到每一个被调用函数，
    观察net的_modules字段何时被构建并且赋值，记录其中所有进入的类与函数

2.  采用sequential容器，改写Alexnet，给features中每一个网络层增加名字，并通过下面这行代码打印出来：
    print(alexnet._modules['features']._modules.keys())
'''

# ================================ 1 ================================

'''
第一步：net = LeNet(classes=2)
第二步：LeNet类，__init__()，super(LeNet, self).__init__()函数
第三步：Module类, __init__()函数/_construct()函数/_setattr()函数
第四步：Conv2d类，__init__()函数
第五步：Module类, __setattr()函数
第六步：Conv2d类，__init__()函数
第七步：Module类, __setattr()函数
第八步：Linear类，__init__()函数
第九步：Module类, _construct()函数/_setattr()函数
第十步：Linear类，__init__()函数
第十一步：Module类, _construct()函数/_setattr()函数
第十二步：Linear类，__init__()函数
第十三步：Module类, _construct()函数/_setattr()函数
第十四步：返回net
'''

# ================================ 2 ================================

import torch
import torchvision
import torch.nn as nn
from collections import OrderedDict


class AlexNetSequentialOrderDict(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNetSequentialOrderDict, self).__init__()
        self.features = nn.Sequential(OrderedDict({
            'conv1' : nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            'relu1': nn.ReLU(inplace=True),
            'pool1': nn.MaxPool2d(kernel_size=3, stride=2),
            'conv2' : nn.Conv2d(64, 192, kernel_size=5, padding=2),
            'relu2': nn.ReLU(inplace=True),
            'pool2': nn.MaxPool2d(kernel_size=3, stride=2),
            'conv3' : nn.Conv2d(192, 384, kernel_size=3, padding=1),
            'relu3': nn.ReLU(inplace=True),
            'conv4' : nn.Conv2d(384, 256, kernel_size=3, padding=1),
            'relu4': nn.ReLU(inplace=True),
            'conv5' : nn.Conv2d(256, 256, kernel_size=3, padding=1),
            'relu5': nn.ReLU(inplace=True),
            'pool3':nn.MaxPool2d(kernel_size=3, stride=2),
        }))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(OrderedDict({
            'dropout1' : nn.Dropout(),
            'fc1': nn.Linear(256 * 6 * 6, 4096),
            'relu6' : nn.ReLU(inplace=True),
            'dropout2' : nn.Dropout(),
            'fc2': nn.Linear(4096, 4096),
            'relu7' : nn.ReLU(inplace=True),
            'fc3': nn.Linear(4096, num_classes),
        }))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    # 原生Alexnet
    alexnet1 = torchvision.models.AlexNet()
    # 修改Alexnet：为网络层增加名字
    alexnet2 = AlexNetSequentialOrderDict()
    print(alexnet2._modules['features']._modules.keys())
