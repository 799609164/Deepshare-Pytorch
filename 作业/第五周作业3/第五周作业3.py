# -*- coding: utf-8 -*-
'''
第五周作业3：
1.  采用torch.nn.Module.register_forward_hook机制实现AlexNet第一个卷积层输出特征图的可视化，
并将/torchvision/models/alexnet.py中第28行改为：nn.ReLU(inplace=False)，观察inplace=True与inplace=False的差异。
'''

# ================================ 1 ================================
'''
inplace为True，会改变input，input=output
inplace为False，则不会改变input，input!=output
'''

import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models


writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")

# 数据
path_img = "./lena.png"     # your path to image
normMean = [0.49139968, 0.48215827, 0.44653124]
normStd = [0.24703233, 0.24348505, 0.26158768]

norm_transform = transforms.Normalize(normMean, normStd)
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    norm_transform])

img_pil = Image.open(path_img).convert('RGB')
if img_transforms is not None:
    img_tensor = img_transforms(img_pil)
img_tensor.unsqueeze_(0)    # chw --> bchw

# 模型
alexnet = models.alexnet(pretrained=False)

# 注册hook
fmap_dict = dict()
for name, sub_module in alexnet.named_modules():

    if isinstance(sub_module, nn.Conv2d):
        key_name = str(sub_module.weight.shape)
        fmap_dict.setdefault(key_name, list())
        n1, n2 = name.split(".")

        def hook_func(m, i, o):
            key_name = str(m.weight.shape)
            fmap_dict[key_name].append(o)

        alexnet._modules[n1]._modules[n2].register_forward_hook(hook_func)

# forward
output = alexnet(img_tensor)

# add image
for layer_name, fmap_list in fmap_dict.items():
    fmap = fmap_list[0]
    fmap.transpose_(0, 1)

    nrow = int(np.sqrt(fmap.shape[0]))
    fmap_grid = vutils.make_grid(fmap, normalize=True, scale_each=True, nrow=nrow)
    writer.add_image('feature map in {}'.format(layer_name), fmap_grid, global_step=322)
