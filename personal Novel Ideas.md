[toc]

# personal Novel Ideas

## 1. 网络结构

### 1.1 from RepVGG and MoE

> - `RepVGG`: 结构重参数化， 训练时采用更多分支，推理时采用重参数化后的单路分支
> - `MoE`: 多专家的一种集成学习或是sparse model（conditional computation）

`Ideas`

> - **设计一些不同分支， 每一层包括多路分支，利用MoE的思想一次只激活其中一些的分支。**
> - **推理时利用结构重参数化加快推理速度。**



### 1.2 transformer 中的MLP 替换

> - 不确定学术中是否有人这么做过，`MLP`可以用卷积替换（1X1 Conv）
> - 那么同样的也可以用其他size 的卷积实现多尺度



### 1.3 MoE的路由 --> 

> - MoE 本身的路由是对其中的不同卷积进行TopK，可以试试将Softmax 的输出直接作用在其网络之后。将不同expert分工协作



##  2. Reference Papers

### 2.1 CondConv

### 2.2  CoE -- With 100M FLOPs Achieving 80%acc@1 in imagenet

[ICML2022](https://arxiv.org/abs/2107.03815)



