# -*- coding:utf-8 -*-
import os
import torch
import time
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tools.my_dataset import RMBDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tools.common_tools import set_seed
from model.lenet import LeNet


set_seed(1)  # 设置随机种子

writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")

split_dir = os.path.join("rmb_split")
train_dir = os.path.join(split_dir, "train")

transform_compose = transforms.Compose([transforms.Resize((32, 64)), transforms.ToTensor()])
train_data = RMBDataset(data_dir=train_dir, transform=transform_compose)
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
data_batch, label_batch = next(iter(train_loader))

img_grid = vutils.make_grid(data_batch, nrow=4, normalize=True, scale_each=True)
writer.add_image("input img", img_grid, 0)

writer.close()

