# ML -- About Math

[toc]

## 1. Similarity computation

### 1.1 Pearson correlation coefficient (PCCs) 与Cosine，Euclidean Distance

>  **皮尔逊相关系数**是用于度量两个变量X和Y之间的相关性（Linear）**其值介于(-1, 1)**
> $$
> \rho(X,Y) = \frac{ E[(X-\mu_X)(Y-\mu_Y)] }{\sigma X\sigma Y}
> $$





> **Cosine Similarity**输出范围也是(-1, 1):
> $$
> c(X,Y) = \frac{X\cdot Y}{|X||Y|} = \frac{\sum_{i=1}^nX_iY_i}{\sqrt\sum_{i=1}^nX_i^2\sqrt\sum_{i=1}^nY_i^2}
> $$



> **Euclidean Distance**: 常见的相似度度量方式，可求两个vector之间的distance，范围为（0，inf）距离越小则相似性越大。欧式距离计算时默认对于每一个维度给予相同weight，可以是用加权欧式距离对取值差距大的维度进行赋权平衡：
> $$
> d(X,y) = \sum_{i=1}^n(X_n - Y_n)^2
> $$



   

- 与Cosine Similarity 在数据被标准化后等价，<font size=4, color=red>即PCCs 是数据经过标准化之后的similarity</font>
- 与欧式距离在标准化数据下等价(成线性关系)

$$
d(X,Y) = 2n(1-\rho(X,Y))
$$



## 2. 拉格朗日



