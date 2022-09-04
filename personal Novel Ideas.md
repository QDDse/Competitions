[toc]

# personal Novel Ideas

## 1. 网络结构

### 1.1 from RepVGG and MoE

> - `RepVGG`: 结构重参数化， 训练时采用更多分支，推理时采用重参数化后的单路分支
> - `MoE`: 多专家的一种集成学习或是sparse model（conditional computation）

`Ideas`

> - **设计一些不同分支， 每一层包括多路分支，利用MoE的思想一次只激活其中一些的分支。**
> - **推理时利用结构重参数化加快推理速度。**