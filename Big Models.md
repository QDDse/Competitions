# Big Models (SY实习notes)、

[TOC]

## Preliminary

> 并行训练：
>
> - DP--- 数据并行： minibatch
>
> - MP---模型并行： Tensor并行， pipeline 并行
> - Pipeline 并行
> - 混合并行： Megatron



> DP -- Data Parallelism
>
> - `Naive DP`: model广播到workers， Data分配（scatter）到workers, 参数同步机制：
>   - `Synchronous`: BSP， 需要等待其他worker
>   - `Asynchronous`: ASP， 异步并行
> - `Bucketing Gradients`: DDP
>   - ![img](https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/pytorch-ddp.png)



> `Model Parallelism`:
>
> - Naive MP:
>   - ![img](https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/naive-data-parallelism.png)
>   - 将model按照layer的顺序分配到workers，但是会造成`GPU利用严重不足`



> `Pipeline Parallelism`
>
> - `Gpipe`
> - ![img](https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/pipedream.png)





## 1. Megatron-ML (Nvidia2019)

### 1.1 inter-layer && intra-layer

- inter-layer: 层间， layer-leve并行
- intra-layer：层内， data-level 并行 

![img](https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/Megatron-LM.png)



## 2. ZeRO ---- (面向万亿级model得Memory optimization)

### 2.1 模型memory usage

> `Model state：`optimizerState、Gradients、 Parameters
>
> `Residual state`: activation 、temporary buffers、 memory fragment



### 2.2 Adam 简介

> `Advantages:`**既能适应稀疏梯度，又能缓解梯度震荡的问题**



### 2.3 内存消耗分析

> - 输入数据：	比较小
> - 模型参数量： 较大
> - 各层响应： 

### Question：



## 3. MoE(Mixture Of Experts)

### 3.1 动机

> - 训练样本越来越大导致训练成本平方级增长
>
> - 提出将大模型拆分成多个小模型，对于一个样本只需`经过激活的一部分小模型`去计算。
> - 引入一个`稀疏门机制`, 样本输入这个gate得到要激活的小模型index。



### 3.2 Challenge

> - GPU 擅长计算不擅长分支
> - 导致小模型样本过少
> - 网络通信
> - 需要在loss上改进以控制稀疏性

### 3.3 Methods

> - 批缩小的问题：
>   - 数据并行与模型并行， 
>   - 单步拆分
>   - 优化模型训练时的内存，进一步增大batchsize



> - 均衡问题

### 3.4 Latest Study

#### 3.4.1 GShard -- 扩展MoE transformer 至600B参数

> - `Sharded MoE` 仅仅将MoE layers 分片到不同的node上，其他layers简单复制
> - `impoved `：
>   - `Expert capacity`:  experts 设置一个阈值（threshold），当token被路由到超过阈值的expert则token被标记 为`overflowed`
>   - `Local group dispatching`“： 
>   - `Auxiliary Loss`: 辅助损失函数，
>   - `Random routing`: 选择第二好的experts的probability 与其权重成正比，否则Gshard遵循随机Routing



### 3.4.2 Switch Transformer -- 扩展到Trillion级别

> `Some Designs`: 
>
> - `Selective Precision`:  fp32精度在router主体使用，results recast 回FP16
> - `Smaller initialization`： 初始化权重s=1 ---> s=0.1
> - `Use higher expert dropout`: Finetune 在小数据集上，为避免overfiting，dropout rate 设置较高，但是在所有layers设置高rate 会导致performance， `dropout rate = 0.1 at non-expert`  &&  `rate = 0.4 at expert FF layers` 
>
> - `Sparse routing`: 不需要稀疏算子可以适应GPU、TPU
> - 每个token只会路由给一个Expert
> - DP、MP、Expert P

### 3.4.3 Export Choice -- 



### 3.4.4 Vision-MoE

[V-MoE repo](https://github.com/google-research/vmoe/tree/main/vmoe)

> `Contribution`：
>
> - 对于`FFN` 进行Router 分配
>
> - 对比Vanille ViT， FLOPS减少的情况下JFT300M@1 acc 以及imagenet5SHot都胜过ViT
>
> - `Batch Priority Routing`:
>   
>   - `Predefined buffer` ： 为每个export预定义一个buffer-size，当分配的tokens超过该export的buffer上限时，drop or not process。更大的buffer capacity会提高性能但是会导致cost。![](https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/AVvXsEgOgAXLqIiYhplKl8FaR9Q6ryDSqzBT4tSlq5gHA41HLjZe0p4KAzLtrLElthgamOwS2Ii07l3fkrZ0MqTUVIRRjwSynWUpglTM7jzu8P8ahewkDp4189puFE4d-hd7UkMALpU0oDAWHyPJg4xBolTw8f7xCisqoa8petv0N0IQcn55cGLYAaRzp16N9Q%253Ds1600.gif)
>   
>     
>   
> - ![](https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/AVvXsEi_t3DCZxIbfi1HHX6bwJ_zDDiT6-tZ1ysVxlq-JVqZ6LUuKRFr4U9ZilxPA86rE4gfwi75ZE4TRzsgcCx9yKHN9NBaB5tkxVLYDP24H48bRuadXbFYJ92rVNGpycKkM_LWigqoWnvHf7yqOi7AHkCvUHgf9YDk6K1Hj8eqAoVnYs13kd_FVXSbC3ZrIw%253Ds670.png)

### 3.4.5 SF-MoE



## MPI通信的方式

### Broadcast

![img](https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/qg6ezsg9va.png)

- 一份相同得data 广播给不同得device

### Scatter

![image.png](https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/3g453iuups.png)

- 可以将不同得data 分发给不同得device



### Gather

![image.png](https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/j8hraqv1fl.png)

- 将不同进程的data拼凑在一起



### Reduce（整合）

![image.png](https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/g76vn718of.png)

![image.png](https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/pa6i3nzijd.png)

- 将多个进程的data 按照given mapping function（映射函数）运算 （F）的结果存在一个进程中，F为<font size=3, color=red>归约操作</font>。 上图中都是sum（求和）
- 操作与ALLreduce相同，但是仅将**结果写入指定跟级别的接受缓冲区**

### All-reduce

![image.png](https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/ai75r9ggk5.png)



- `reduce  + Broadcast` ： 将不同进程的归约结果发送到这些进程中，`reduce+broadcast`可以看作是all-reduce的一种实现。

### All-gather --- 强调整合在一起

![image-20220719111140542](https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/image-20220719111140542.png)



![image-20220719155535958](https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/image-20220719155535958.png)

>K 个处理器中的每一个来自各个处理进程N个值聚合成维度为K*N的输出，输出按照rank排序





## Workspace

### 1. Tools 之 Einsum， Einops

