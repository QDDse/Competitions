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



### 2. [Megatrion-2]（Sequence Parallel）

[paper](https://arxiv.org/pdf/2205.05198.pdf)

> 

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
>   - 辅助loss function

### 3.4 Latest Study -- (现有的框架)

> - **Meta** ： `fairseq` , `BASE Layers`
> - **Google**: `GShard`
> - **Tsinghua-U** : `FasterMoE ` （FastMoE的改进）
> - **Microsoft**： `TUTEL`

#### 3.4.1 GShard -- 扩展MoE transformer 至600B参数

[]()

> - `Sharded MoE` 仅仅将MoE layers 分片到不同的node上，其他layers简单复制
> - `impoved `：
>   - `Expert capacity`:  experts 设置一个阈值（threshold），当token被路由到超过阈值的expert则token被标记 为`overflowed`
>   - `Local group dispatching`“： 
>   - `Auxiliary Loss`: 辅助损失函数，
>   - `Random routing`: 选择第二好的experts的probability 与其权重成正比，否则Gshard遵循随机Routing



### 3.4.2 Switch Transformer -- 扩展到Trillion级别

[Switch]: https://arxiv.org/abs/2101.03961

[Simple Achieve](https://nn.labml.ai/transformers/switch/index.html)



> `Some Designs`: 
>
> - `Selective Precision`:  fp32精度在router主体使用，results recast 回FP16
> - `Smaller initialization`： 初始化权重s=1 ---> s=0.1
> - `Use higher expert dropout`: Finetune 在小数据集上，为避免overfiting，dropout rate 设置较高，但是在所有layers设置高rate 会导致performance， `dropout rate = 0.1 at non-expert`  &&  `rate = 0.4 at expert FF layers` 
>
> - `Sparse routing`: 不需要稀疏算子可以适应GPU、TPU
> - 每个token只会路由给一个Expert
> - DP、MP、Expert P

### 3.4.3 Export Choice -- 改进传统的Token-Choice

[`paper`](https://arxiv.org/abs/2202.09368)

> - `Token-Choice`：
>   - 传统MoE（Convenient MoE) 通过Token选择Top-K experts进行路由
>   - 会导致**sub-optimal trainig** 的缺陷
> - `Load-Imbalance`: 负载不均衡，传统MoE通过辅助loss function（auxiliary loss），但是并不能很好保证负载均衡；同时辅助函数与主损失进行加权，增加了模型的复杂性。
> - `Under Specialization`: trade-off of load-balance and specialization
> - `Same Compute for tokens`: 每个token都选择fixed number expert，而不能根据输入的复杂性进行调整。

### 3.4.4 Vision-MoE

[V-MoE repo](https://github.com/google-research/vmoe/tree/main/vmoe)

> `Contribution`：
>
> - 对于`FFN` 进行Router 分配
> - 对比Vanille ViT， FLOPS减少的情况下JFT300M@1 acc 以及imagenet5SHot都胜过ViT
> - `Batch Priority Routing`:
>
>   - `Predefined buffer` ： 为每个export预定义一个buffer-size，当分配的tokens超过该export的buffer上限时，drop or not process。更大的buffer capacity会提高性能但是会导致cost。
>   - 对token 进行重要性排序，按照重要性大小舍弃掉分数不高得tokens.
>   - ![](https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/AVvXsEgOgAXLqIiYhplKl8FaR9Q6ryDSqzBT4tSlq5gHA41HLjZe0p4KAzLtrLElthgamOwS2Ii07l3fkrZ0MqTUVIRRjwSynWUpglTM7jzu8P8ahewkDp4189puFE4d-hd7UkMALpU0oDAWHyPJg4xBolTw8f7xCisqoa8petv0N0IQcn55cGLYAaRzp16N9Q%253Ds1600.gif)
>
> 
>
> - ![](https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/AVvXsEi_t3DCZxIbfi1HHX6bwJ_zDDiT6-tZ1ysVxlq-JVqZ6LUuKRFr4U9ZilxPA86rE4gfwi75ZE4TRzsgcCx9yKHN9NBaB5tkxVLYDP24H48bRuadXbFYJ92rVNGpycKkM_LWigqoWnvHf7yqOi7AHkCvUHgf9YDk6K1Hj8eqAoVnYs13kd_FVXSbC3ZrIw%253Ds670.png)

`capacity ratio` :signal_strength:定义为capacity / tokens recived per device



### 3.4.5 SF-MoE 

### 3.4.6 Swin-MoE（Microsoft） TUTEL 

[Swin-transformer](https://github.com/microsoft/Swin-Transformer)

> - TUTEl 由微软建立并于fairseq toolkit进行合并
> - TUTEL 实现了Swin-TransformerV2 MoE
> - `Contributions`:
>   -  by adaptive methods for dynamic MoE workload at any scale.
>   - adaptive parallelism switching and adaptive pipelining
>   - 2-dimention Hierarchical all-to-all algorithm

![image-20220825103738672](https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/image-20220825103738672.png)

### 3.4.7 Deep-Speed MoE --- （PR-MoE， MoS）

> - `Challenges of Convient MoE`:
>   - `Limited Scope`: MoE目前只能应用在encoder-decoder，seq2seq tasks。
>   - `Massive Memory Requirements`：alse called `parameter efficiency`, 通常一个MoE 稀疏模型是dense-counterparts 10x的参数量。T5-model可以在单卡V100 32G fit，MT-NLG（500B）对应的sparse model理论上需要5k GPUs
>   - `INference Performance`: 推理同样需要10x以上的带宽（memory bandwidth）， 推理延迟（inference latency）更加依赖于model-szie，memory-bandwidth，从memory中读取数据的capabilities。

> - `Solutions`:
>
>   - 扩展MoE至NLG tasks
>
>   - `PR-MoE && MoS`: 
>
>     - `PR-MoE`: Pyramid-Residual MoE，只在最有效的地方应用MoE
>
>     ![image-20220822164908539](https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/image-20220822164908539.png)
>
>     - `MoS`: 知识蒸馏版的PR-MoE, 减少了3.7x的model size 同时保持了comparable model quality



### 3.4.8 LIMoE --- (Multimodal the Language Image Mixture of Experts)

[LIMoE](https://ai.googleblog.com/2022/06/limoe-learning-multiple-modalities-with.html) [Arxiv](https://arxiv.org/pdf/2206.02770.pdf)

![](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgSZvvfrRuopj58bPE0B7bGxdeJKRbksIrdWOdF_MC4N2fndx6-rfEt0E9IzI4Qag_sTtGm9vFzv-2y_iOOLHhu2oFJ9lRK-lqpyypFBow3cz9beeDHu6Ji9ePTQcWv6B6RSWUZG6CBqkbLuG_JGm7nehPnMQ_sqtcUGH5_ll6K5sOvMXfvC1zCZ_Y9qg/s1600/image3.gif)

​		                                      **LIMoE Architecture**

> `Contributions`:
>
> - `New auxiliary loss`
> - `BPR`: Routing prioritization <font color=red>(非本文提出)</font>



### 3.4.9 [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://ai.googleblog.com/2021/12/more-efficient-in-context-learning-with.html)

[paper](https://arxiv.org/abs/2112.06905)  `Generalist Language Model`: 通用的语言MoE模型

> - `对比GPT-3`， 都是decoder only model
> - 构建了一个Trillion-level dataset，从网上爬取的文本利用训练好的filter 进行过滤挑选除高质量的text
> - `Parameters`： 1.2T （64 experts per layer and 32 MoE layers）
> - 激活97B（8% of 1.2T）
> - [`GSPMD`](https://ai.googleblog.com/2021/12/general-and-scalable-parallelization.html):  本文中用到该architecture
> - `Expert's subset size` 2 
> - `relative positional bias` 

![img](https://blogger.googleusercontent.com/img/a/AVvXsEhKmA36iEaxIj0bSi7IUfctim-muB1B695evFlPNoqkiIbD_mVkkbdslk6HNwEXCBJmGsvxyDYsNTS7-YzZlblcQa2St498_RPwm72uz6jrK8mooPA6-wN4vqNswF-MG8JEPsgj7R4Rsm60Xh1xVLQtSeYrfhNEG4pr-7MbTHYdxjbug845Fy5smz17ZA=w402-h640)



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



### AlltoAll

![image-20220823111002003](https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/image-20220823111002003.png)

> `P0`包含[0,1,2,3,4], 将0赋值给自己，1传给P1，2传给P2....
>
> - All-to-All 定制通讯也叫做全部交换，常用于各种并行算法中，如快速傅里叶，矩阵变换。。



## torch.distributed

- `distributed.init_process_group(): 多个进程同步并通信，建立多进程分布式group`
  - `world_size`: 总的进程数
- `nn.utils.data.DistributedSampler`: 为各个进程切分数据以保证数据non-overlap

## Workspace

### 1. Tools 之 Einsum， Einops

### 2. Vision MoE

### 3 阅读Megaton-LM code

> - 阅读megatron code


