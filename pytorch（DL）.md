[TOC]



# pytorch（DL）

## 1.1 一般步骤

```python
### 1.模型定义
```



```python
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 卷积
        self.pool = nn.MaxPool2d(2, 2) # 池化
        self.conv2 = nn.Conv2d(6, 16, 5) # 卷积
        # 全链接层，最后是输出10分类
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```



### 2.数据加载处理

继承 torch.utils.Dataset

### 3.training the module

> 定义网络
>
> 定义数据
>
> 定义损失函数和优化器
>
> 开始训练
>
> >训练网络
> >
> >> 将梯度set 0
> >
> >> 求出loss
> >
> >> backforward
> >
> >> update parameters
> >
> >> update Learning rate

>> 可视化指标
>>
>> 计算在验证集上的指标

```python
# 定义网络
net = Net()

# 定义数据
#数据预处理，1.转为tensor，2.归一化
transform = transforms.Compose(    
     [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# 训练集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
# 验证集
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

# 定义损失函数和优化器 
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 开始训练
net.train()
for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # 将梯度置为0
        # zero the parameter gradients
        optimizer.zero_grad()
        # 求loss
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # 梯度反向传播
        loss.backward()
        # 由梯度，更新参数
        optimizer.step()

        # 可视化
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

# 查看在验证集上的效果
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

net.eval()
outputs = net(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
```





### 4.Test the module

> 训练的参数：
>
> >dataset parameters :  文件路径、batch_size
> >
> >training parameters: 学习率、epoch
> >
> >module parameters: 输入大小、输出大小

### 5.训练过程可视化(optional)

### 6. Dataset download

- ~~~              python
  import torchvision
  train_set = torchvision.MNIST(root='location',train=True,download=True)
  test_set = torchvision.MNIST(root='location',train=False,download=True) 
  ~~~

  

## 1.2 搭建神经网络的方式

### 1.2.1 nn.Sequential

~~~py
model = torch.nn.Sequential(
torch.nn.Linear(10,20),
torch.nn.ReLU(),
torch.nn.Linear(20,2),
)
~~~

### 1.2.2 继承nn.Module

~~~py
import torch
class SalaryNet(torch.nn.Module):
	def __init__(self,in_size,h1_size,h2_size,out_size):
		super(SalaryNet,self).__init__()
		self.h1  = torch.nn.Linear(in_size,h1_size)
		self.relu = torch.nn.ReLU()
		self.h2 = torch.nn.Linear(h1_size,h2_size)
		self.out = torch.nn.Linear(h2_size,out_size)
	
	def forward(self,x):
	h1_relu = self.relu(self.h1(x))
	h2_relu = self.relu(self.h2(h1_relu))
	predict = self.out(h2_relu)
	return predict
~~~



## 1.5 处理多维度的输入

~~~ python
self.linear = torch.nn.Linear(8,2) # 8维空间（features数量） 降至2维
~~~

- Epoch，Batch-size，Iterations

  - Epoch = batch*Iterations

  - Epoch为所有数据的一次训练

  - ![image-20211022174458220](C:\Users\强大大\AppData\Roaming\Typora\typora-user-images\image-20211022174458220.png)

  - ~~~python
    import torch
    import torch.utils.data import Dataset  # Dataset为抽象类 不能直接实例化
    import torch.utils.data import DataLoader
    class DiabeteDataset(Dataset):
        def __init__(self):
            pass
        def __getitem__(self,index):
            pass
        def __len__(self):  # Magic方法返回length 
            pass
    dataset  = DiabeteDataset()  # 子类继承Dataset后进行实例化
    train_loader = DataLoader(dataset=dataset,
                             batch_size = 32,shuffle = True,
                             num_worker = 2)
        
    ~~~

 

# 2. ML常见的概念

## 2.1 缺失值的处理

- 均值填充
- 中位数填充
- 非空数据训练一个模型预测缺失值的特征

## 2.2 Tensor(张量 )

| Rank |                             fig                              |   Name    |   名称   |
| :--: | :----------------------------------------------------------: | :-------: | :------: |
|  0   | ![image-20211005090815827](C:\Users\强大大\AppData\Roaming\Typora\typora-user-images\image-20211005090815827.png) |  Scalar   |   标量   |
|  1   | ![image-20211005090847212](C:\Users\强大大\AppData\Roaming\Typora\typora-user-images\image-20211005090847212.png) |  Vector   |   向量   |
|  2   | ![image-20211005090913727](C:\Users\强大大\AppData\Roaming\Typora\typora-user-images\image-20211005090913727.png) |  Matrix   |   矩阵   |
|  3   | ![image-20211005090949959](C:\Users\强大大\AppData\Roaming\Typora\typora-user-images\image-20211005090949959.png) | 3D Array  | 三阶张量 |
| >=4  | ![image-20211005091045587](C:\Users\强大大\AppData\Roaming\Typora\typora-user-images\image-20211005091045587.png) | N-D Array | N阶张量  |

## 2.3 数据标准化和数据正则化

### 2.3.1 数据标准化 （归一化）

-  Z-Score 

  - 对序列做出变化

  $$
  yi = (xi-X')/S
  $$

  - pytorch实现

    ~~~python
    import torch
    import matplolib.pyplot as plt
     
    ~~~


## 2.4 python中的常用DS

- 列表List ，字典dict --mutable
-  元组tutle ，集合set

### 2.4.1 list

1. 列表中的每个元素都***可变***的，意味着可以对每个元素进行修改和删除；
2. 列表是有序的，每个元素的位置是确定的，可以用索引去访问每个元素；
3. 列表中的元素可以是Python中的任何对象；

~~~py
mylist = ['Google','Yahoo','Baidu']  # [] 并用，分割
mylist[0] = 'Microsoft'
mylist.append('Alibaba') #后面添加元素
mylist.insert(1,'Tencent')  # 在指定位置插入元素
mylist.pop(1)  #删除1 index的元素，并返回该元素
mylist.remove('Tencent')  # delete
del mylist[1:3]  # 删除索引在1-3的元素

~~~

## 2.5 广播机制

### 2.5.1 自动广播规则

- 每个维度的大小都一样

  ~~~
  x = torch.ones(1,2,3)
  y = torch.zeros(1,2,3)
  print((x+y).shape)
  ~~~

- 每个tensor至少有一个维度

~~~ python
x = torch.Tensor([2])
y = torch.Tensor([3],[5])
print(x+y)
# tensor([[5.],
		   [7.]])
~~~

- 两个Tensor的对应维度的大小应满足三个条件之一

  - 对应相等

  - 其中一个tensor的大小等于1

  - 其中一个tensor的某个维度不存在

  - ~~~python
    #从里向外的维度 
    # 7 == 7 ; 1,1维度为一；4 == 4；3 == 3；维度缺失
    x = torch.randn(2,3,4,5,6,7)
    y = torch.randn(3,4,1,1,7)
    print((x+y).shape)
    ~~~

    

## 2.6 GPU设备及并行编程

### 2.6.1 device 和cuda.device

### 2.6.2 从 cpu到gpu

- copy_，to，cuda等方法





# 3.torch常用的package

## 3.1 四类包

- 类似于`numpy`的 通用数组包
- `torch.autograd`：用于构建计算图形并自动获取渐变的包
- `torch.nn`：具有共同层和成本函数的神经网络库
- `torch.optim`：具有通用优化算法（如SGD，Adam等）的优化包
- `torch.functional as F`

### 3.1.1 torch.optim

1. SDG,RMSprop,Adam

- ~~~python
  # SGD 就是随机梯度下降
  opt_SGD         = torch.optim.SGD(net_SGD.parameters(), lr=LR)
  # momentum 动量加速,在SGD函数里指定momentum的值即可
  opt_Momentum    = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
  # RMSprop 指定参数alpha
  opt_RMSprop     = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
  # Adam 参数betas=(0.9, 0.99)
  opt_Adam        = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
  ~~~

### 3.4.1 Functional

1. `torch.nn.Conv2d` 于 `torch.functional.conv2d`

> - 对于nn.Conv 可以理解为简单实现卷积（类操作）
> - F.conv2d 为自定义式卷积，函数时接口，可以自定义卷积核
>
> ~~~py
> import torch
> import torch.functional as F
> W = nn.Parameters(torch.Tensor(K,C,H,W))
> B = nn.Parameters(torch.Tensor(K))
> out = F.conv2d(inp, W, B, stride, padding, group)
> ~~~
>
> 

## 3.2 备用函数

### 3.2.1 iloc与loc

- loc 按照index名取值

- iloc按照行数（num）取值

- ~~~
  print(data.loc['A'])
  print(data.iloc[0])
  
  ~~~


### 3.2.2 format

- ~~~python
  print('{name}在{option}'.format(name = 'pan',option = 'sleeping'))  
  # pan在sleeping
  print('{name} is {option}'.format('pan','/'))
  
  ~~~

- 

### 3.2.3 torch.linspace

- torch.linspace(start, end, steps=100, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)

- 返回从start 到 end 数量为steps的等距的一维Tensor.

### 3.2.4 torch.squeeze（） 

- ~~~py
  torch.squeeze(1)  # 只对为1的维度进行删减压缩；
  torch.unsqueeze（1）  # 同上进行扩展；
  ~~~

- ​       

### 3.2.5 zip（）函数

- 将对象中对应的元素打包成tuple（元组）

- ~~~ python
  a = [1,2,3]
  b = [4,5,6]
  zipped = zip(a,b)
  # zipped = [(1,4),(2,5),(3,6)]
  ~~~

### 3.2.6 创建矩阵

- torch.rand  （均匀分布的随机矩阵）   
- torch.randn（标准正态分布的随机矩阵）
- torch.randperm（） （创建一个随机的整数序列）
- torch.arange（列表）
- torch.zeros（）（元素为0的矩阵）

### 3.2.7 数据持久化与高并发

- torch.save  与 torch.load

- ~~~python
  # 序列化模型
  x = torch.randn(2,3)
  # 序列化  sava方法
  torch.save(x,"randn")
  # 反序列化  load 方法
  x_load = torch.load("randn")
  ~~~

- 并发设计框架

- ~~~python
  # pytorch 的默认线程数量等于计算机内核个数
  threads = torch.get_num_threads()
  print(threads)
  # 设置并发数
  torch.set_num_threads(4)
  threads_1 = torch.get_num_threads()
  print(thread_1)
  ~~~

- 

# * Tips

## 1.one-hot 编码

- one-hot 编码：[0 0 0 0 0 0 0 1 0 ]只有一个为1其余为0的的状态。

- 可以借助Pandas等数据处理工具。

- ~~~py
  import pandas as pd
  df = pd.read_csv("salary.csv",encoding = 'utf-8')
  df_x1 = pd.get_dummies()
  ~~~
  

## 2. 维度诅咒

- 维度 N 则采样越多x的N次方，成本越高

## 3.visdom —可视化package

 

## 4. why batch

- MSE 均值平方和的 loss 可以采用并行计算时间复杂度小，但是收敛性能差
- SGD 随机梯度下降法 收敛性好 ，但是无法进行并行计算导致时间复杂度高。
- 因此  将批量数据分成batch 组，中和两种算法的优点。

## 5.加载pth或训练权重

### 5.1 .pth文件

- 有序字典（包含data，requires_grad)

~~~python
import torch
state_dict = torch.load("resnet18.pth")
print(type(state_dict))
~~~

### 5.2 torch.save()

~~~python
state_dict = {'net' : model.state_dict(),'optimizer' : optimizer.state_dict(),'epoch':epoch
				}
torch.save(state_dict,dir)
torch.save(model.state_dict,dir)
~~~

### 5.3 torch.load()

- 恢复某一阶段的训练（或者Test） ，读取之前保存的checkpoint 的model parameter.

~~~python
checkpoint = torch.load(dir)
model.load_state_dict(check['net'])
optimizer.load_state_dict(check['optimizer'])
state_epoch  = checkpoint['epoch']+1
~~~

### 5.3 加载pre-train model部分的参数

~~~python
resnet = models.resnet50(pretrained = True)
new_state_dict = resnet.state_dict()
dd = net.state_dict()  # net is the model that includes the resnet backbone
for k in new_state_dict.keys():
    print(k)
    if k in dd.keys() and not k.startswith('fc')：# 不加载全连接参数
    	print("yes")
        dd[k] = new_state_dict[k]
net.load_state_dict(dd)
~~~

## 6 Module

### 6.1 nn.LSTM()

~~~py
import torch
import torch.nn as nn
# 定义一个双向LSTM
bilstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2,bidirection=True)
input = torch.randn(5, 3, 10)
h0 = torch.randn(4, 3, 20)
c0 = torch.randn(4, 3, 20)
output,(hn,cn) = bilstm(input, (h0,c0))
# output(5,3,40) ----双向输出为每个time step正逆cat起来
# hn----->num_layers*num_direction, batch, hidden_size
~~~

## 7. 生成对抗

### 7.1 威胁模型

### 7.2 FGSM(Fast Gradient Sign Attack)

- 攻击是利用损失函数的梯度，然后调整输入数据以最大化损失。

## 8. Hooks

**用于打印中间层输出输入**

~~~py
import torch
import torch.nn as nn


class TestForHook(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_1 = nn.Linear(in_features=2, out_features=2)
        self.linear_2 = nn.Linear(in_features=2, out_features=1)
        self.relu = nn.ReLU()
        self.relu6 = nn.ReLU6()
        self.initialize()

    def forward(self, x):
        linear_1 = self.linear_1(x)
        linear_2 = self.linear_2(linear_1)
        relu = self.relu(linear_2)
        relu_6 = self.relu6(relu)
        layers_in = (x, linear_1, linear_2)
        layers_out = (linear_1, linear_2, relu)
        return relu_6, layers_in, layers_out

    def initialize(self):
        """ 定义特殊的初始化，用于验证是不是获取了权重"""
        self.linear_1.weight = torch.nn.Parameter(torch.FloatTensor([[1, 1], [1, 1]]))
        self.linear_1.bias = torch.nn.Parameter(torch.FloatTensor([1, 1]))
        self.linear_2.weight = torch.nn.Parameter(torch.FloatTensor([[1, 1]]))
        self.linear_2.bias = torch.nn.Parameter(torch.FloatTensor([1]))
        return True

# 1：定义用于获取网络各层输入输出tensor的容器
# 并定义module_name用于记录相应的module名字
module_name = []
features_in_hook = []
features_out_hook = []


# 2：hook函数负责将获取的输入输出添加到feature列表中
# 并提供相应的module名字
def hook(module, fea_in, fea_out):
    print("hooker working")
    module_name.append(module.__class__)
    features_in_hook.append(fea_in)
    features_out_hook.append(fea_out)
    return None

# 3：定义全部是1的输入
x = torch.FloatTensor([[0.1, 0.1], [0.1, 0.1]])

# 4:注册钩子可以对某些层单独进行
net = TestForHook()
net_chilren = net.children()
for child in net_chilren:
    if not isinstance(child, nn.ReLU6):
        child.register_forward_hook(hook=hook)

# 5:测试网络输出
out, features_in_forward, features_out_forward = net(x)
print("*"*5+"forward return features"+"*"*5)
print(features_in_forward)
print(features_out_forward)
print("*"*5+"forward return features"+"*"*5)


# 6:测试features_in是不是存储了输入
print("*"*5+"hook record features"+"*"*5)
print(features_in_hook)
print(features_out_hook)
print(module_name)
print("*"*5+"hook record features"+"*"*5)

# 7：测试forward返回的feautes_in是不是和hook记录的一致
print("sub result")
for forward_return, hook_record in zip(features_in_forward, features_in_hook):
    print(forward_return-hook_record[0])
~~~



# quick-lookup

## 定义model

~~~python
class net(nn.Module):
    def __init__(self, *params):
        super(net, self).__init__()
        self.params = [*params]
        pass
    def forward(self, *input):
        return 
    def 
    
~~~

 

## 并行训练

~~~python
## 数据并行
torch.nn.DataParallel()
## 多进程接口
torch.distrbuted()
## 搭配多进程接口
torch.multiprocessing()
~~~

> <font size=5, color=orange>**分布式训练**</font>
>
> 
>
> ` torch.distributed` 提供了通用分布式系统常见的概念：
>
> - `group`: 进程组
> - `world`: Global process number
> - `rank`: 进程序号
> - `local_rank`: 进程内的GPU编号
>
> ~~~python
> ## 训练demo：
> ugpus_per_node = torch.cuda.device_count()
> 
> ## 分布式初始化
> torch.distributed.init_process_group(
>     backend = 'nccl'  ## GPU   or  'gloo' in cpu
>     init_method = 'tcp://127.0.0.1:8009'  ## one node multi-gpu 推荐使用tcp通信
>     world_size = xx,
>     rank = xx
> )
> ## 在多卡时选择master用于打印信息和其他操作
> def main_worker(gpu, args):
>     
>     # code...
> 	### 这里选择的是rank=0的gpu
>     if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
>         args.master = True
>     else:
>         args.master = False
> ~~~



> <font size=5, color=orange>训练数据处理</font>
>
> - Pytorch分布式训练是将data直接切分成`world_size`份，然后在各个进程内独立处理数据。
>
> ```python
> if torch.distributed.is_initialized():
>     train_sampler = torch.utils.data.distributed.DistributedSampler()
> ```
>
> 





### 参数部分

####  1.`argparse` 利用命令行进行传参

```python
## test argparser
import argparse
from ast import arg

from parso import parse
parser = argparse.ArgumentParser()

parser.add_argument('--device', default='0, 1', type=str, help='设置使用哪些GPU')
parser.add_argument('--no_cuda', action='store_true', help='不适用GPU训练')

args = parser.parse_args(args=[])  ## ipynb中的格式
args = parser.parse_args()         ## py中的格式
```

> TIPS:
>
> - py文件中`parser.parse_args()`即可
> - ipynb文件中因为`sys.argsv[1:]本身具有两个默认值，导致parser error`





### inplace operation

* Pytorch 中inplace 操作属于原地操作，对Tensor operate时不copy另外的内存，而是直接在原始内存修改

* Pytorch 中用`_`表示 

  * ~~~python
    ## demo
    .add_()
    .transpose_()...
    ~~~

  * 

