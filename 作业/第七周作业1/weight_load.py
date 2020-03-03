import model
import torch


net_new = model.LeNet2(classes=2)

# 加载模型参数
path_state_dict = "./model_state_dict.pkl"
state_dict_load = torch.load(path_state_dict)

# 权重赋值
print("加载前: ", net_new.features[0].weight[0, ...])
net_new.load_state_dict(state_dict_load)
print("加载后: ", net_new.features[0].weight[0, ...])