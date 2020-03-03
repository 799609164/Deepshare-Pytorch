import model
import torch


net = model.LeNet2(classes=2)

# 权重赋值
print("训练前: ", net.features[0].weight[0, ...])
net.initialize()
print("训练后: ", net.features[0].weight[0, ...])

# 保存模型参数
path_state_dict = "./model_state_dict.pkl"
net_state_dict = net.state_dict()
torch.save(net_state_dict, path_state_dict)