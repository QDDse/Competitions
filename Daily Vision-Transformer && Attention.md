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
>           self.dim = dim
>             self.num_heads = num_heads
>           head_dim = dim // num_heads
> 

![image-20220810143459304](https://raw.githubusercontent.com/QDDse/MD_images/main/image-20220810143459304.png)



> - `Stage1` patch=4
> - `stage2` patch=2

### 1.2 CoAtNet -- 结合CNN与Transformer

### 1.3 Shunted Transformer

[`**知乎**`](https://zhuanlan.zhihu.com/p/450244412)

> `Novelty`：
>
> - `MTA`在单个attention计算时得到multi-scaleKV，再计算selfattention-
> - ![img](https://raw.githubusercontent.com/QDDse/MD_images/main/53980a60f2f64f8ba771a3f261f78adf.png)
> - ~~~python
>   class Attention(nn.Module):
>       def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
>           super().__init__()
>           assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
>           self.dim = dim
>           self.num_heads = num_heads
>           head_dim = dim // num_heads
>           self.scale = qk_scale or head_dim ** -0.5
>           self.q = nn.Linear(dim, dim, bias=qkv_bias)
>           self.attn_drop = nn.Dropout(attn_drop)
>           self.proj = nn.Linear(dim, dim)
>           self.proj_drop = nn.Dropout(proj_drop)
>           self.sr_ratio = sr_ratio
>           if sr_ratio > 1:
>               self.act = nn.GELU()
>               if sr_ratio==8:
>                   self.sr1 = nn.Conv2d(dim, dim, kernel_size=8, stride=8)
>                   self.norm1 = nn.LayerNorm(dim)
>                   self.sr2 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
>                   self.norm2 = nn.LayerNorm(dim)
>               if sr_ratio==4:
>                   self.sr1 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
>                   self.norm1 = nn.LayerNorm(dim)
>                   self.sr2 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
>                   self.norm2 = nn.LayerNorm(dim)
>               if sr_ratio==2:
>                   self.sr1 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
>                   self.norm1 = nn.LayerNorm(dim)
>                   self.sr2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
>                   self.norm2 = nn.LayerNorm(dim)
>               self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
>               self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
>               self.local_conv1 = nn.Conv2d(dim//2, dim//2, kernel_size=3, padding=1, stride=1, groups=dim//2)
>               self.local_conv2 = nn.Conv2d(dim//2, dim//2, kernel_size=3, padding=1, stride=1, groups=dim//2)
>           else:
>               self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
>               self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
>           self.apply(self._init_weights)
<<<<<<< HEAD
>                             
=======
>                           
>>>>>>> ad0da59062fa85d637becb5fb1c04049da34836f
>       def forward(self, x, H, W):
>           B, N, C = x.shape
>           q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
>           if self.sr_ratio > 1:
>           x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
>           x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))
>           x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))
>           kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4)
>           kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4)
>           k1, v1 = kv1[0], kv1[1] #B head N C
>           k2, v2 = kv2[0], kv2[1]
>           attn1 = (q[:, :self.num_heads//2] @ k1.transpose(-2, -1)) * self.scale
>           attn1 = attn1.softmax(dim=-1)
>           attn1 = self.attn_drop(attn1)
>           v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C//2).
>                      transpose(1, 2).view(B,C//2, H//self.sr_ratio, W//self.sr_ratio)).\
>           view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2)
>           x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C//2)
>           attn2 = (q[:, self.num_heads // 2:] @ k2.transpose(-2, -1)) * self.scale
>           attn2 = attn2.softmax(dim=-1)
>           attn2 = self.attn_drop(attn2)
>           v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C//2).
>                      transpose(1, 2).view(B, C//2, H*2//self.sr_ratio, W*2//self.sr_ratio)).\
>           view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2)
>           x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C//2)
>
> 
>
> - `Detail Specofic`: 即**DW-Conv**
>
>   - ![img](https://raw.githubusercontent.com/QDDse/MD_images/main/ada64f45205645d2b0ef165d8b988451.png)
>
>
> 
>
> 





### 1.4 STN（Spatial Transformer）

> - (1) STN作为一种独立的模块可以在不同网络结构的任意节点插入任意个数并具有运算速度快的特点,它几乎没有增加原网络的运算负担,甚至在一些attentive model中实现了一定程度上的加速。
> - (2) STN模块同样使得网络在训练过程中学习到如何通过空间变换来减少损失函数,使得模型的损失函数有着可观的减少。
> - (3) STN模块决定如何进行空间变换的因素包含在Localisation net以及之前的所有网络层中。
> - (4) 网络除了可以利用STN输出的Feature map外,同样可以将变换参数作为后面网络的输入,由于其中包含着变换的方式和尺度,因而可以从中得到原本特征的某些姿势或角度信息等。
> - (5) 同一个网络结构中,不同的网络位置均可以插入STN模块,从而实现对与不同feature map的空间变换。
> - (6) 同一个网络层中也可以插入多个STN来对于多个物体进行不同的空间变换,但这同样也是STN的一个问题:由于STN中包含crop的功能,所以往往同一个STN模块仅用于检测单个物体并会对其他信息进行剔除。同一个网络层中的STN模块个数在一定程度上影响了网络可以处理的最大物体数量。

~~~python
## 首先定义Localisation net的特征提取
self.localization = nn.Sequential(
    nn.Conv2d(1,8,kernel_size=7),
    nn.MaxPool2d(2, stride=2),
    nn.ReLU(True),
    nn.Conv2d(8,10,kernel_size=5),
    nn.MaxPool2d(2, stride=2),
    nn.ReLU(True)
)
## 定义Localization net的参数回归部分
self.fc_loc = nn.Sequential(
    nn.Linear(10*3*3, 32),
    nn.ReLU(True),
    nn.Linear(32, 3*2)  # Affine transform 输出为 3*2
)
## 在nn.Module中定义完整的STN
def stn(self, x):
    xs = self.Localization(x)
    xs = xs.view(-1, 10*3*3)
    theta = self.fc_loc(xs)
    theta = theta.view(-1, 2, 3)
    grid = F.affine_grid(theta, x.size())
    x = F.grid_sample(x, grid)
    return x

## 完整的toy model with STN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net().to(device)

## About Transformer

### 1. Position Encoding && Position Embedding

- `Position encoding`: from Vanille ViT :

~~~python
## 实现sinusoid position vec
def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            # this part calculate the position In brackets
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        # [:, 0::2] are all even subscripts, is dim_2i
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)
~~~

- `Position Embedding`: Each position of the sequence will be mapped to a trainable vec of size ==dim==

~~~python
## init abs position
pos_embID = torch.nn.Parameter(torch.randn(max_seq_tokens, dim))
## during forward pass
input_to_trm_mhsa = input_embedding + pos_embID[:current_seq_tokens, :]

out = transformer(input_to_trm_mhsa)

## relative position emb
import torch
import torch.nn as nn
from einops import rearrange

# borrowed from lucidrains
#https://github.com/lucidrains/bottleneck-transformer-pytorch/blob/main/bottleneck_transformer_pytorch/bottleneck_transformer_pytorch.py#L21
def relative_to_absolute(q):
    """
    Converts the dimension that is specified from the axis
    from relative distances (with length 2*tokens-1) to absolute distance (length tokens)
      Input: [bs, heads, length, 2*length - 1]
      Output: [bs, heads, length, length]
    """
    b, h, l, _, device, dtype = *q.shape, q.device, q.dtype
    dd = {'device': device, 'dtype': dtype}
    col_pad = torch.zeros((b, h, l, 1), **dd)
    x = torch.cat((q, col_pad), dim=3)  # zero pad 2l-1 to 2l
    flat_x = rearrange(x, 'b h l c -> b h (l c)')
    flat_pad = torch.zeros((b, h, l - 1), **dd)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
    final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
    final_x = final_x[:, :, :l, (l - 1):]
    return final_x


### 1.5 Hornet---- `gnConv`
~~~python
## 官方实现 GnConv2d
### GnConv 本身是sequence2sequence module： 输入输出为同维度
### 可以insert to model中间层
def get_dwconv(dim, kernel, bias):
    return nn.Conv2d()
class gnconv(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]  ## dim,dim/2,dim/4,dim/8,dim/16
        self.dims.reverse()                                ## dim/16.dim/8,dim/4,dim/2,dim
        self.proj_in = nn.Conv2d(dim, 2*dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)
        
        self.proj_out = nn.Conv2d(dim, dim, 1)
        ## dim/16-->dim/8.... point-wise Conv
        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i+1], 1) for i in range(order-1)]
        )

        self.scale = s
        print('[gnconv]', order, 'order with dims=', self.dims, 'scale=%.4f'%self.scale)

    def forward(self, x, mask=None, dummy=False):
        B, C, H, W = x.shape

        fused_x = self.proj_in(x) ## C-->2C
        ## dims=[1,2,4,8,16] if dim == 16
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
        ## pwa:(B,dim/16, H,W) 
        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]  # (B, dim/16, H,W) *

        for i in range(self.order -1):
            x = self.pws[i](x) * dw_list[i+1]

        x = self.proj_out(x) 

        return x
~~~

### 1.5 Focal Transformer ----- (NIPS2021)

> - [`知乎`](https://zhuanlan.zhihu.com/p/417106453)
>
> - [`Arxiv`](https://arxiv.org/pdf/2107.00641.pdf)



### 1.6 Swin-Transformer (MoE)

[code](https://github.com/microsoft/Swin-Transformer/)

- `torchinfo_summary`:

<img src="https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/image-20220907172748560.png" alt="image-20220907172748560" style="zoom:67%;" />

> - `SW-Attention`
>
>   - 第一步: `torch.roll`,tensor移动
>   - 第二部： `img_mask` -- > `attn_mask`,位置相同则work_attn, 不同则not work
>
>   <img src="https://pic2.zhimg.com/v2-fe69e3e4c8d9cb8d7092da8c249201fd_r.jpg" alt="preview" style="zoom: 33%;" />
>
>   <img src="https://pic4.zhimg.com/v2-392d9d14fc64026e0c0b97ead5f2ada3_r.jpg" alt="preview" style="zoom:33%;" />
>
>   <img src="https://pic1.zhimg.com/80/v2-1ed6987c14ee88781811844140def1d4_1440w.jpg" alt="img" style="zoom:33%;" />
>
>   - `key code`:
>
> ~~~py
> import torch
> 
> import matplotlib.pyplot as plt
> 
> 
> def window_partition(x, window_size):
>     """
>     Args:
>         x: (B, H, W, C)
>         window_size (int): window size
> 
>     Returns:
>         windows: (num_windows*B, window_size, window_size, C)
>     """
>     B, H, W, C = x.shape
>     x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
>     windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
>     return windows
> 
> 
> window_size = 7
> shift_size = 3
> H, W = 56, 56
> img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
> h_slices = (slice(0, -window_size),
>             slice(-window_size, -shift_size),
>             slice(-shift_size, None))
> w_slices = (slice(0, -window_size),
>             slice(-window_size, -shift_size),
>             slice(-shift_size, None))
> cnt = 0
> for h in h_slices:
>     for w in w_slices:
>         img_mask[:, h, w, :] = cnt
>         cnt += 1
> 
> mask_windows_ = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
> print(mask_windows_.shape)
> mask_windows = mask_windows_.view(-1, window_size * window_size)
> 
> attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # (nW, 1, win_size^2) - (nW, win_size^2, 1)
> attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
> 
> plt.matshow(img_mask[0, :, :, 0].numpy())
> plt.matshow(attn_mask[0].numpy())
> plt.matshow(attn_mask[1].numpy())
> plt.matshow(attn_mask[2].numpy())
> plt.matshow(attn_mask[3].numpy())
> 
> plt.show()
> ## 同一个region内的token彼此attention是work， 但是不同region的token之间不work！
> ~~~
>
> 

<img src="https://user-images.githubusercontent.com/42901638/115496492-5d5e6f80-a29c-11eb-8cbf-76aff3e40d7e.png" alt="image" style="zoom:67%;" />





### 1.7 Cross ViT

[paper](https://arxiv.org/abs/2103.14899)

### 1.8 DeiT

[paper](https://arxiv.org/abs/2012.12877)[知乎](https://zhuanlan.zhihu.com/p/344085679)



### 1.9 Going deeper ViT

[paper](https://arxiv.org/abs/2103.17239)





### 1.10 T2T_ViT

[paper](https://arxiv.org/abs/2101.11986)[知乎](https://zhuanlan.zhihu.com/p/359930253)



### 1.11 Deepvit

[paper](https://arxiv.org/abs/2103.11886)

### 1.12 TimeSformer

[paper](https://arxiv.org/abs/2102.05095)

### 1.13 Mobile ViT

[paper](https://arxiv.org/abs/2110.02178) [github](https://github.com/apple/ml-cvnets)  [Repo](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/MobileViT)



<img src="https://img-blog.csdnimg.cn/73aa3a646e0447fa9f00b0a673a95ddb.png" alt="img" style="zoom:150%;" />



### 1.14 HorNet ---NIPS22

[paper](https://arxiv.org/pdf/2207.14284.pdf) [Code](https://github.com/raoyongming/HorNet)

![image-20220919213421505](https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/image-20220919213421505.png)

​                                                                         HorNet 结构

>
> 



# Self-Supervisied Learning

## 2. MIM (Mask Image Model)

### 2.1 BEiT (BERT in CV)

[paper](https://arxiv.org/abs/2106.08254) [github](https://github.com/microsoft/unilm/tree/master/beit)

[paper_v2](https://arxiv.org/abs/2208.06366)

[paper_3](https://arxiv.org/pdf/2208.10442.pdf)

> - 该github 目前集成了`BeiT`, `BEiTv2`, `HuggingFace`
> - 

------



# Litter in CV

## 1. RepVGG -- (Reparameterize && Inception)



## 2. Diffusion Model



