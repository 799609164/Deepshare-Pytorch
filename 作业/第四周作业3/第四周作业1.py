# -*- coding: utf-8 -*-
'''
第四周作业3：
1.  优化器的作用是管理并更新参数组，请构建一个SGD优化器，通过add_param_group方法添加三组参数，
    三组参数的学习率分别为0.01，0.02，0.03，momentum分别为0.9， 0.8， 0.7，构建好之后，
    并打印优化器中的param_groups属性中的每一个元素的key和value（提示：param_groups是list，其每一个元素是一个字典）
'''

# ================================ 1 ================================

import torch
import torch.optim as optim


w1 = torch.randn((2, 2), requires_grad=True)
w2 = torch.randn((2, 2), requires_grad=True)
w3 = torch.randn((2, 2), requires_grad=True)

optimizer = optim.SGD([w1], lr=0.01, momentum=0.9)

optimizer.add_param_group({'params': w2, 'lr':0.02, 'momentum':0.8})
optimizer.add_param_group({'params': w3, 'lr':0.03, 'momentum':0.7})

#print("optimizer.param_groups is\n{}".format(optimizer.param_groups))
for index, group in enumerate(optimizer.param_groups):
    params = group['params']
    lr = group['lr']
    momentum = group['momentum']
    print("第【{}】组参数 params 为:\n{} \n学习率 lr 为:{} \n动量 momentum 为:{}".format(index, params, lr, momentum))
    print("==============================================")
