[TOC]

# Daily Vision-Transformer && Attention

## 1 Transformer backbone

### 1.1 PVT-(Pyramid ViT)

> `Novelty` ： 
>
> - 将每一stage的patch与dim 按照CNN的层级进行排布
>
> - 为较少计算量： 用Spatial-reduction attention 代替 MHA， 减少了KQ的数量
>
> - ![preview](https://raw.githubusercontent.com/QDDse/MD_images/main/v2-744faf9810100dde9718a53b90d61023_r.jpg)
>
> - ~~~python
>   class Attention(nn.Module):
>       def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
>           super(Attention, self).__init__()
>           assert dim % num_heads == 0
>           
>           self.dim = dim
>           self.num_heads = num_heads
>           head_dim = dim // num_heads
>           
>           self.scale = qk_scale or head_dim ** -0.5
>           self.q = nn.Linear(dim, dim, bias=qkv_bias)
>           self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
>           self.attn_drop = nn.Dropout(attn_drop)
>           self.proj = nn.Linear(dim ,dim)
>           self.proj_drop = nn.Dropout(proj_drop)
>           self.sr_ratio = sr_ratio
>           ## conv
>           if sr_ratio > 1:
>               self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
>               self.norm = nn.LayerNorm(dim)
>           
>       def forward(self, x, H, W):
>           B, N, C = x.shape
>           q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
>           ## einops更加优雅
>           # q = rearrange(self.q(x), 'b n (h c) -> b h n c', h=self.num_heads)
>           if self.sr_ratio > 1:
>               x_ = x.permute(0,2,1).reshape(B, C, H, W)
>               x_ = self.sr(x_).reshape(B, C, -1).permute(0,2,1)
>               x_ = self.norm(x_)
>               kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
>               print('kv_shape:{}'.format(kv.shape))
>           else:
>               kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
>               print('kv_shape:{}'.format(kv.shape))
>           k, v = kv[0], kv[1]  # (B, H, n, c)
>           
>           attn = (q @ k.transpose(-2, -1)) * self.scale
>           print('attn_shape:{}'.format(attn.shape))
>           attn = attn.softmax(dim=-1)
>           attn = self.attn_drop(attn)
>           
>           x = (attn @ v).transpose(1,2).reshape(B, N, C)
>           print('output_shape:{}'.format(x.shape))
>           x = self.proj(x)
>           return self.proj_drop(x)
>   ~~~

![image-20220810143459304](https://raw.githubusercontent.com/QDDse/MD_images/main/image-20220810143459304.png)



> - `Stage1` patch=4
> - `stage2` patch=2

### 1.2 CoAtNet -- 结合CNN与Transformer

### 1.3 Shunted Transformer

> `Novelty`：
>
> - `MTA`在单个attention计算时得到multi-scaleKV，再计算selfattention-
> - ![img](https://raw.githubusercontent.com/QDDse/MD_images/main/53980a60f2f64f8ba771a3f261f78adf.png)







