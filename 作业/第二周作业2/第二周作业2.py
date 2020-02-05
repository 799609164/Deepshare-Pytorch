# -*- coding: utf-8 -*-
'''
第二周作业2：
1.  将介绍的transforms方法实现对图片的变换，通过plt.savefig将图片保存下来
    不少于10张不一样的数据增强变换的图片，如裁剪，缩放，平移，翻转，色彩变换，错切，遮挡等等
    
2.  自定义一个增加椒盐噪声的transforms方法，使得其能正确运行
'''

import os
import numpy as np
import torch
import random
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from my_dataset import RMBDataset
from common_tools import transform_invert

# ================================ 1 ================================

def homework1(test_dir):
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # 1 裁剪
        #transforms.CenterCrop(120),
        # 2 翻转
        #transforms.RandomHorizontalFlip(p=1),
        # 3 旋转
        #transforms.RandomRotation(45),
        # 4 色相
        #transforms.ColorJitter(hue=0.4),
        # 5 饱和度
        #transforms.ColorJitter(saturation=50),
        # 6 灰度图
        #transforms.Grayscale(3),
        # 7 错切
        #transforms.RandomAffine(0,shear=45),
        # 8 缩放
        #transforms.RandomAffine(0,scale=(0.5,0.5)),
        # 9 平移
        #transforms.RandomAffine(0,translate=(0.5,0)),
        # 10 遮挡
        #transforms.ToTensor(),
        #transforms.RandomErasing(p=0.5,scale=(0.1,0.4),value=0),

        transforms.ToTensor(),
    ])
    # 构建MyDataset实例
    test_data = RMBDataset(data_dir=test_dir, transform=test_transform)
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    for i, data in enumerate(test_loader):
        inputs, labels = data   # B C H W
        img_tensor = inputs[0, ...]     # C H W
        img = transform_invert(img_tensor, test_transform)
        plt.imshow(img)
        plt.show()
        plt.pause(0.5)
        plt.close()

# ================================ 2 ================================

class AddPepperNoise(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) or (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255   # 盐噪声
            img_[mask == 2] = 0     # 椒噪声
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img


def homework2(test_dir):
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        AddPepperNoise(0.9, p=0.8),
        transforms.ToTensor(),
    ])
    test_data = RMBDataset(data_dir=test_dir, transform=test_transform)
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    for i, data in enumerate(test_loader):
        inputs, labels = data   # B C H W
        img_tensor = inputs[0, ...]     # C H W
        img = transform_invert(img_tensor, test_transform)
        plt.imshow(img)
        plt.show()
        plt.pause(0.5)
        plt.close()


if __name__ == '__main__':
    test_dir = './data/'
    #homework1(test_dir)
    homework2(test_dir)
