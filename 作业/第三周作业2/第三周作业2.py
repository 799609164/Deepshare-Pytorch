# -*- coding: utf-8 -*-
'''
第三周作业2：
1.  深入理解二维卷积，采用手算的方式实现以下卷积操作，然后用代码验证
    (1)采用2个尺寸为3*3的卷积核对3通道的5*5图像进行卷积，padding=0，stride=1，dilation=0
    其中，input_shape = (3, 5, 5)，kernel_size = 3*3， 
    第一个卷积核所有权值均为1， 第二个卷积核所有权值均为2，
    计算输出的feature_map尺寸以及所有像素值

    数据：[[1,1,1,1,1],     [[2,2,2,2,2],       [[3,3,3,3,3],
          [1,1,1,1,1],      [2,2,2,2,2],        [3,3,3,3,3],
          [1,1,1,1,1],      [2,2,2,2,2],        [3,3,3,3,3],
          [1,1,1,1,1],      [2,2,2,2,2],        [3,3,3,3,3],
          [1,1,1,1,1]]      [2,2,2,2,2]]        [3,3,3,3,3]]
    
    (2)接(1)，上下左右四条边均采用padding，padding=1，填充值为0，计算输出的feature map尺寸以及所有像素值

2.  对lena图片进行(3*3*3)3d卷积，提示：padding=(1, 0, 0)
'''

# ================================ 1 ================================

import torch
import torch.nn as nn

# (1) output_shape = ( 5 - 3 ) / 1 + 1 = 3
input_tensor = torch.tensor([[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],
                            [[2,2,2,2,2],[2,2,2,2,2],[2,2,2,2,2],[2,2,2,2,2],[2,2,2,2,2]],
                            [[3,3,3,3,3],[3,3,3,3,3],[3,3,3,3,3],[3,3,3,3,3],[3,3,3,3,3]]], dtype=torch.float)
input_tensor.unsqueeze_(dim=0)

conv_layer1 = nn.Conv2d(3, 1, 3, bias=False)
conv_layer1.weight.data = torch.ones(conv_layer1.weight.shape)

conv_layer2 = nn.Conv2d(3, 1, 3, bias=False)
conv_layer2.weight.data = torch.full(conv_layer2.weight.shape,2)

output_tensor_1 = conv_layer1(input_tensor)
print("卷积前尺寸:{}\n卷积后尺寸:{}" .format(input_tensor.shape, output_tensor_1.shape))
print("像素值:", output_tensor_1)
print("=======================================")
output_tensor_2 = conv_layer2(input_tensor)
print("卷积前尺寸:{}\n卷积后尺寸:{}" .format(input_tensor.shape, output_tensor_2.shape))
print("像素值:", output_tensor_2)

# (2)
conv_layer3 = nn.Conv2d(3, 1, 3, bias=False, padding=1)
conv_layer3.weight.data = torch.ones(conv_layer3.weight.shape)

conv_layer4 = nn.Conv2d(3, 1, 3, bias=False, padding=1)
conv_layer4.weight.data = torch.full(conv_layer4.weight.shape,2)

print("=======================================")
output_tensor_3 = conv_layer3(input_tensor)
print("卷积前尺寸:{}\n卷积后尺寸:{}" .format(input_tensor.shape, output_tensor_3.shape))
print("像素值:", output_tensor_3)
print("=======================================")
output_tensor_4 = conv_layer4(input_tensor)
print("卷积前尺寸:{}\n卷积后尺寸:{}" .format(input_tensor.shape, output_tensor_4.shape))
print("像素值:", output_tensor_4)

# ================================ 2 ================================

import matplotlib.pyplot as plt
from common_tools import transform_invert
from torchvision import transforms
from PIL import Image


lena_img = "lena.png"
img = Image.open(lena_img).convert('RGB')
img_transform = transforms.Compose([transforms.ToTensor()])
img_tensor = img_transform(img)
img_tensor.unsqueeze_(dim=0)

conv_layer = nn.Conv3d(3, 1, (3, 3, 3), padding=(1, 0, 0), bias=False)
nn.init.xavier_normal_(conv_layer.weight.data)
img_tensor.unsqueeze_(dim=2)
img_conv = conv_layer(img_tensor)

#img_conv = transform_invert(img_conv[0, 0:1, ...], img_transform)
img_conv = transform_invert(img_conv[:, :, ...], img_transform)
img_raw = transform_invert(img_tensor.squeeze(), img_transform)
plt.subplot(122).imshow(img_conv, cmap='gray')
plt.subplot(121).imshow(img_raw)
plt.show()
