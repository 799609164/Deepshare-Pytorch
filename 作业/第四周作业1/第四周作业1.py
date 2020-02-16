# -*- coding: utf-8 -*-
'''
第四周作业1：
1.  Lossfunction依旧属于网络层的概念，即仍旧是Module的子类，为了对lossfunction有一个更清晰的概念，
    需要大家采用步进(Step into)的调试方法从loss_functoin = nn.CrossEntropyLoss()语句进入函数，
    观察从nn.CrossEntropyLoss()到class Module(object)一共经历了哪些类，记录其中所有进入的类及函数

2.  损失函数的reduction有三种模式，它们的作用分别是什么？
    当inputs和target及weight分别如以下参数时，reduction='mean'模式时，loss是如何计算得到的？
    
    inputs = torch.tensor([[1, 2], [1, 3], [1, 3]], dtype=torch.float)
    target = torch.tensor([0, 1, 1], dtype=torch.long)
    weights = torch.tensor([1, 2])
'''

# ================================ 1 ================================

'''
第一步：CrossEntropyLoss类，super(CrossEntropyLoss, self).__init__
第二步：_WeightedLoss类，__init__()函数
第三步：_Loss类，__init__()函数
第四步：进入Module类
'''

# ================================ 2 ================================

'''
none：逐个元素计算
sum：所有元素求和
mean：加权平均
'''
# 加权平均：总loss/(1+2+2)
import torch
import torch.nn as nn


inputs = torch.tensor([[1, 2], [1, 3], [1, 3]], dtype=torch.float)
target = torch.tensor([0, 1, 1], dtype=torch.long)

weights = torch.tensor([1, 2], dtype=torch.float)

loss_f_mean = nn.CrossEntropyLoss(weight=weights, reduction='mean')
loss_mean = loss_f_mean(inputs,target)
print("Cross Entropy Loss:\n ", loss_mean)
