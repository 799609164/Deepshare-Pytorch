# 第四周

## 1. 权值初始化

### 1.1 梯度消失与爆炸

$$
H_2 = H_1 \times W_2
$$

$$
\begin{aligned}{\Delta W_{2}}=\frac{\partial Loss}{\partial out} \times \frac{\partial o u t}{\partial H_{2}} \times \frac{\partial H_{2}}{\partial w_{2}} \\=\frac{\partial Loss}{\partial out} \times \frac{\partial o u t}{\partial H_{2}} \times H_1
\end{aligned}
$$

$\Delta W_2$ 的梯度取决于上一层的输出 $H_1$

梯度消失：

$$
\begin{aligned}
&\mathrm{H}_{1} \rightarrow 0 \Rightarrow \Delta \mathrm{W}_{2} \rightarrow 0 
\end{aligned}
$$

梯度爆炸：

$$
\begin{aligned}
&\mathrm{H}_{1} \rightarrow \infty \Rightarrow \Delta \mathrm{W}_{2} \rightarrow \infty
\end{aligned}
$$

![](https://raw.githubusercontent.com/799609164/Gallery/master/DeepShare_pytorch/4_1.png)

### 1.2 Xavier初始化

**Xavier初始化**

![](https://raw.githubusercontent.com/799609164/Gallery/master/DeepShare_pytorch/4_2.png)

**Kaiming初始化**

![](https://raw.githubusercontent.com/799609164/Gallery/master/DeepShare_pytorch/4_3.png)

其中，$a$ 为负半轴斜率

### 1.3 权值初始化方法

* Xavier均匀分布
* Xavier标准正态分布
* Kaiming均匀分布
* Kaiming标准正态分布
* 均匀分布
* 正态分布
* 常数分布
* 正交矩阵初始化
* 单位矩阵初始化
* 稀疏矩阵初始化

```python
# 计算激活函数的方差变换尺度
nn.init.calculate_gain(nonlinearity,    # 激活函数名称
        param=None)    # 激活函数参数，如Leaky ReLU的negative_slop
```

## 2. 损失函数

### 2.1 损失函数概念

* **损失函数**：衡量模型输出与真实标签的差异
* 损失函数（Loss Function）：$Loss = f(y^\prime, y)$
* 代价函数（Cost Function）：$Cost=\frac{1}{N} \sum_{i}^{N} f\left(y_{i}^{\prime}, y_{i}\right)$
* 目标函数（Objective Function）：$Obj=Cost+Regularization$（L1/L2··正则项）

```python
class _Loss(Module):
    def __init__(self,reduction='mean'):
        super(_Loss, slef).__init__()
        self.reduction = reduction
```

### 2.2 交叉熵损失函数

![](https://raw.githubusercontent.com/799609164/Gallery/master/DeepShare_pytorch/4_4.png)

* 相对熵：KL散度
* $P$ 为训练集数据分布；$Q$ 为模型输出数据分布

```python
# 计算交叉熵
nn.CrossEntropyLoss(weight=None,    # 各类别loss设置的权值
            ignore_index=-100,    # 忽略某个类别
            reduction='mean')    # 计算模式，如none/sum/mean
            # none：逐个元素计算
            # sum：所有元素求和，返回标题
            # mean：加权平均，返回标量
```

### 2.3 NLL/BCE/BCEWithLogits Loss

```python
# 1. 实现负对数似然函数中的负号功能
nn.NLLLoss(weight=None,    # 各类别的loss设置权值
        ignore_index=-100,    # 忽略某个类别
        reduction='mean')    # 计算模式，如none/sum/mean
# 2. 二分类交叉熵
# 注：输入值为[0,1]，可配合sigmoid函数使用
nn.BCELoss(weight=None,    # 各类别loss设置的权值
        ignore_index=-100,    # 忽略某个类别
        reduction='mean')    # 计算模式，如none/sum/mean
# 3. 结合Sigmoid与二分类交叉熵
# 注：不需要额外加入Sigmoid函数
nn.BCEWithLogitsLoss(weight=None,    # 各类别loss设置的权值
            ignore_index=-100,    # 忽略某个类别
            reduction='mean',    # 计算模式，如none/sum/mean
            pos_weight=None)    # 正样本的权值
```

### 2.4 其他损失函数

1. `nn.L1Loss`：$l_n=|x_n-y_n|$

1. `nn.MSELoss`：$l_n=(x_n-y_n)^2$

1. `nn.SmoothL1Loss`：$loss(x,y)=\frac{1}{n} \sum_{i=1}^{n}{z_i}$

    其中，$z_{i}=\left\{\begin{array}{ll}{0.5\left(x_{i}-y_{i}\right)^{2},} & {\text { if }\left|x_{i}-y_{i}\right|<1} \\ {\left|x_{i}-y_{i}\right|-0.5,} & {\text { otherwise }}\end{array}\right.$

1. `nn.PoissonNLLLoss`

1. `nn.KLDivLoss`：$l_n=y_n \times (logy_n-x_n)$

1. `nn.MarginRankingLoss`：$loss(x,y)=max(0,-y \times (x_1-x_2)+margin)$

1. `nn.MultiLabelMarginLoss`：$\operatorname{loss}(x, y)=\sum_{ij} \frac{\max (0,1-(x[y(j])-x[i]))}{x.size(0)}$

1. `nn.SoftMarginLoss`：$\operatorname{loss}(x, y)=\sum_{i} \frac{\log (1+\exp (-y[i] * x[i]))}{x.nelement()}$

1. `nn.MultiLabelSoftMarginLoss`：$\operatorname{loss}(x, y)=-\frac{1}{x.nelement()} * \sum_{i} y[i] * \log \left((1+\exp (-x[i]))^{-1}\right)+(1-y[i]) * \log \left(\frac{\exp (-x|i|)}{(1+\exp (-x[i)))}\right)$

1. `nn.MultiMarginLoss`：$\operatorname{loss}(x, y)=\frac{\left.\sum_{i} \max (0, \operatorname{margin}-x[y]+x[i])\right)^{p}}{x.size(0)}$

1. `nn.TripletMarginLoss`：$L(a,p,n)=max\{d(a_i,p_i)-d(a_i,n_i)+\text{margin},0\}$
    其中，$d(x_i,y_i)={||x_i-y_i||}_p$    

1. `nn.HingeEmbeddingLoss`：$l_{n}=\left\{\begin{array}{ll}{x_{n},} & {\text { if } y_{n}=1} \\ {\max \left\{0, \text{margin}-x_{n}\right\},} & {\text { if } y_{n}=-1}\end{array}\right.$

1. `nn.CosineEmbeddingLoss`：$\operatorname{loss}(x, y)=\left\{\begin{array}{ll}{1-\cos \left(x_{1}, x_{2}\right),} & {\text { if } y=1} \\ {\max \left(0, \cos \left(x_{1}, x_{2}\right)-\text { margin }\right),} & {\text { if } y=-1}\end{array}\right.$

1. `nn.CTCLoss`（新添加）

```python
# 1. nn.L1Loss
# 计算inputs与target之差的绝对值
nn.L1Loss(reduction='mean')    # 计算模式，如none/sum/mean
# 2. nn.MSELoss
# 计算inputs与target之差的屏方
nn.MSELoss(reduction='mean')
# 3. SmoothL1Loss
# 平滑的L1Loss
nn.SmoothL1Loss(reduction='mean')
# 4. PoissonNLLLoss
# 泊松分布的负对数似然损失函数
# log_input = True：loss(input,target)=exp(input)-target*input
# log_input = False：loss(input,target)=input-target*log(input+eps)
nn.PoissonNLLLoss(log_input=True,    # 输入是否为对数形式
            full=False,    # 是否计算所有loss
            eps=1e-08,    # 修正项，避免log为nan
            reduction='mean')
# 5. KLDivLoss
# 计算KL散度，相对熵
# 注：需提前输入计算log-probabilities，如通过nn.logsoftmax()计算
# batchmean：在batchsize维度求平均值
nn.KLDivLoss(reduction='mean')    # 计算模式增加batchmean
# 6. MarginRankingLoss
# 计算两个向量之间的相似度，用于排序任务，返回一个n*n的loss矩阵
# y=1时，当x1>x2，不产生loss；y=-1时，当x2>x1，不产生loss
nn.MarginRankingLoss(margin=0,    # 边界值，x1与x2之间的差异值
            reduction='mean')
# 7. MultiLabelMarginLoss
# 多标签边界损失函数
# 例：四分类任务/样本属于0类和3类，标签为[0,3,-1,-1]
nn.MultiLabelMarginLoss(reduction='mean') 
# 8. SoftMarginLoss
# 计算二分类的logistic损失
nn.SoftMarginLoss(reduction='mean')
# 9. MultiLabelSoftMarginLoss
# SoftMarginLoss多标签版本，标签为[1,0,0,1]
nn.MultiLabelSoftMarginLoss(weight=None,    # 各类别的loss设置权值
            reduction='mean')
# 10. MultiMarginLoss
# 计算多分类的折页损失
nn.MultiMarginLoss(p=1,    # 可选1或2
            margin=1.0,    # 边界值
            weight=None,    # 各类别的loss设置权值
            reduction='mean')
# 11. TripletMarginLoss
# 计算三元组损失，人脸验证中常用
nn.TripletMarginLoss(margin=1.0,    # 边界值
            p=2,    # 范数的阶
            eps=1e-06,
            swap=False,
            reductionn='mean')
# 12. HingeEmbeddingLoss
# 计算两个输入的相似性，常用于非线性embedding和半监督学习
# 注：输入x为两个输入之差的绝对值
nn.HingeEmbeddingLoss(margin=1.0,
            reduction='mean')
# 13. CosineEmbeddingLoss
# 采用余弦相似度计算两个输入的相似性
nn.CosineEmbeddingLoss(margin=0,    # 可取值[-1,1]，推荐[0,0.5]
            reductionn='mean')
# 14. CTCLoss
# 计算CTC损失，解决时序类数据的分类
nn.CTCLoss(blank=0,    # blank label
        reduction='mean',    # 无穷大的值或梯度置0
        zero_infinity=False)
```

![](https://raw.githubusercontent.com/799609164/Gallery/master/DeepShare_pytorch/4_5.png)

### 2.5 作业

[第四周作业1](https://github.com/799609164/Deepshare-Pytorch/tree/master/%E4%BD%9C%E4%B8%9A/%E7%AC%AC%E5%9B%9B%E5%91%A8%E4%BD%9C%E4%B8%9A1)

## 3. 优化器

### 3.1 优化器的概念

* 优化器：管理并更新模型中可学习参数的值，使得模型输出更接近真实标签
* 导数：函数在指定坐标轴上的变化率
* 方向导数：函数在指定方向上的变化率
* 梯度：向量，方向为方向导数取得最大值的方向

### 3.2 优化器的属性

```python
class Optimizer(object):
    def __init__(self, params, defaults):
        self.defaults = defaults    # 优化器超参数
        self.state = defaultdict(dict)    # 参数的缓存，如momentum的缓存
        # param_groups = [{'params': param_groups}]
        self.param_grops = []    # 管理的参数组，包含字典元素的list
        # _step_count：记录更新次数，学习率调整中使用
        ...
```

### 3.3 优化器的方法

* `zero_grad()`：清空管理参数的梯度
    注：pytorch中张量梯度不自动清零
* `step()`：执行一步更新
* `add_param_group`：添加参数组
* `state_dict()`：获取优化器当前状态信息字典
* `load_state_dict()`：加载状态信息字典

## 4. 随机梯度下降

### 4.1 learning rate 学习率

* 控制更新的步伐

### 4.2 momentum 动量

* 结合当前梯度与上一次更新信息，用于当前更新

* 指数加权平均：$v_t=\beta \times v_{t-1}+(1-\beta)\times \theta_t$

### 4.3 torch.optim.SGD

$$
v_i=m \times v_{i-1}+g(w_i)
$$

$$
w_{i+1}=w_i-lr \times v_i
$$

其中，$w_{i+1}$ 为第 $i+1$ 次更新的参数；$lr$ 为学习率；$v_i$ 为更新量；$m$ 为momentum系数；$g(w_i)$ 为 $wi$ 的梯度

```python
torch.optim.SGD(params,    # 管理的参数组
        lr=<object object>,    # 初始学习率
        momentum=0,    # 动量系数beta
        dampening=0,    # L2正则化系数
        weight_decay=0,
        nesterov=False)    # 是否采用NAG
```

### 4.4 Pytorch的十种优化器

* `optim.SGD`：随机梯度下降法
* `optim.Adagrad`：自适应学习率梯度下降法
* `optim.RMSprop`：Adagrad的改进
* `optim.Adadelta`：Adagrad的改进
* `optim.Adam`：RMSprop结合Momentum
* `optim.Adamax`：Adam增加学习率上限
* `optim.SparseAdam`：稀疏版的Adam
* `optim.ASGD`：随机平均梯度下降
* `optim.Rprop`：弹性反向传播
* `optim.LBFGS`：BFGS的改进

### 4.5 作业

[第四周作业3](https://github.com/799609164/Deepshare-Pytorch/tree/master/%E4%BD%9C%E4%B8%9A/%E7%AC%AC%E5%9B%9B%E5%91%A8%E4%BD%9C%E4%B8%9A3)

注：作业2为整理损失函数笔记