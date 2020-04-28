# 第七周


## 1. 模型保存与加载

### 1.1 序列化与反序列化

* 序列化：将内存中的对象以二进制的形式保存在硬盘中

```python
# 1. 序列化
torch.save(obj,    # 序列化对象
        f)    # 输出路径
# 2. 反序列化
torch.load(f,    # 文件路径
        map_location)    # CPU/GPU
```

### 1.2 保存与加载的两种方式

* 保存

```python
# 1. 保存整个模型
torch.save(net, path)
# 2. 保存模型参数
state_dict = net.state_dict()
torch.save(state_dict, path)
```

* 加载

```python
# 1. 加载整个模型
net = torch.load(path,
        map_location)    # 指定存放的位置，CPU/GPU
# 2. 保存模型参数
state_dict_load = torch.load(path)
print(state_dict_load.keys())
# 将模型参数加载到新的网络中
new_net.load_state_dict(state_dict_load)
```

### 1.3 模型断点续训练

```python
# 1. 保存
checkpoint = {"model_state_dict":net.state_dict(),
            "optimizer_stat_dict":optimizer.state_dict(),
            "epoch":epoch}
torch.save(check_point, path)
# 2. 加载
check_point = torch.load(path)
net.load_state_dict(check_point['model_state_dict'])
optimizer.load_state_dict(check_point['optimizer_state_dict'])
start_epoch = check_point['epoch']
scheduler.last_epoch = start_epoch
```

## 2. 模型微调

### 2.1 迁移学习与模型微调

* 迁移学习：研究源域的知识如何应用到目标域中

### 2.2 PyTorch中的模型微调

* 步骤
    1. 获取预训练参数
    2. 加载模型（load_state_dict）
    3. 修改输出层
* 训练方法
    1. 固定预训练的参数（requires_grad=False/lr=0）
    2. 特征提取器使用较小的学习率（params_group）

### 2.3 作业

[第七周作业1](https://github.com/799609164/Deepshare-Pytorch/tree/master/%E4%BD%9C%E4%B8%9A/%E7%AC%AC%E4%B8%83%E5%91%A8%E4%BD%9C%E4%B8%9A1)

## 3. GPU的使用

### 3.1 CPU与GPU

* CPU(Central Processing Unit 中央处理器)：包括控制器与运算器
* GPU(Graphics Processing Unit 图形处理器)：处理统一的，无依赖的大规模数据运算

![7_1](https://raw.githubusercontent.com/799609164/Gallery/master/DeepShare_pytorch/7_1.png)

### 3.2 数据迁移至GPU

```python
# 1. 数据迁移至GPU
data.to("cuda")
# 2. 数据迁移至CPU
data.to("cpu")
```

* `data` 两种类型
    * Tensor
    * Module

```python
# 例1 使用to函数转换数据类型
# 1.
x = torch.ones((3,3))
x = x.to(torch.float64)
#2.
linear = nn.Linear(2,2)
linear.to(torch.double)
# 例2 使用to函数转换数据设备
# 1.
x = x.to("cuda")
# 2.
gpu1 = torch.device("cuda")
linear.to(gpu1)
```

* **注：** 张量不执行 inplace，模型执行 inplace

### 3.3 多GPU并行运算

* `torch.cuda` 的常用方法

```python
# 1. 计算当前可见可用GPU数
torch.cuda.device_count()
# 2. 获取GPU名称
torch.cuda.get_device_name()
# 3. 为当前GPU设置随机种子
torch.cuda.manual_seed()
# 4. 为所以可见可用GPU设置随机种子
torch.cuda.manual_seed_all()
# 5. 设置主GPU为哪一个物理GPU
# 不推荐
torch.cuda.set_device()
# 推荐：可以更好的分配GPU
# 逻辑GPU0与GPU1 对应 物理GPU2与GPU3，如下图所示
os.environ.setdefault("CUDA_VISIBLE_DEVICES","2,3")
```

![7_2](https://raw.githubusercontent.com/799609164/Gallery/master/DeepShare_pytorch/7_2.png)

* 多GPU运算的分发并行机制：分发-并行运算-结果回收

```python
# 包装模型，实现分发并行机制
torch.nn.DataParallel(module,    # 需要包装分发的模型
        device_ids=None,    # 可分发的GPU，默认分发到所有可见可用GPU
        output_device=None,    # 结果输出设备
        dim=0)
```

* 查询当前GPU剩余内存

```python
def get_gpu_memory():
    import os
    os.system("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp.txt")
    memory_gpu = [int(x.split()[2]) for x in open('tmp.txt', 'r').readlines()]
    os.system('rm tmp.txt')
    return memory_gpu
```

使用案例：

![7_3](https://raw.githubusercontent.com/799609164/Gallery/master/DeepShare_pytorch/7_3.png)


### 3.4 GPU使用的常见报错

![7-4](https://raw.githubusercontent.com/799609164/Gallery/master/DeepShare_pytorch/7_4.png)

![7-5](https://raw.githubusercontent.com/799609164/Gallery/master/DeepShare_pytorch/7_5.png)

## 4. PyTorch常见报错

[PyTorch常见错误与坑汇总文档](https://shimo.im/docs/PvgHytYygPVGJ8Hv)
