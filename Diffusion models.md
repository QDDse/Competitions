[toc]

# Diffusion models

## 1. Metric -- ref（https://zhuanlan.zhihu.com/p/432965561）

> `FID` (**Fréchet inception distance** )

- 直接考虑generated imgs 与true imgs在feature-level的distance
- FID使用Inception Net-V3 fc前的2048维向量作为图片的features
- 公式：

$$
FID = ||\mu_r - \mu_g||^2 + Tr(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})

$$

- \Sigma --- 协方差矩阵
- Tr--- 迹
