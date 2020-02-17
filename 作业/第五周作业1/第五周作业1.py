# -*- coding: utf-8 -*-
'''
第五周作业1：
1.  熟悉TensorBoard的运行机制，安装TensorBoard，并绘制曲线 y=2*x
'''

# ================================ 1 ================================

# 安装：pip install tensorboard
# 运行：tensorboard --logdir=./runs

import numpy as np
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter(comment='y=2x')

for x in range(100):
    writer.add_scalar('y=2x', 2*x, x)

writer.close()
