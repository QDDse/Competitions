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
>      
>       def forward(self, x, H, W):
>           B, N, C = x.shape
>           q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
>           if self.sr_ratio > 1:
>                   x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
>                   x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))
>                   x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))
>                   kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4)
>                   kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4)
>                   k1, v1 = kv1[0], kv1[1] #B head N C
>                   k2, v2 = kv2[0], kv2[1]
>                   attn1 = (q[:, :self.num_heads//2] @ k1.transpose(-2, -1)) * self.scale
>                   attn1 = attn1.softmax(dim=-1)
>                   attn1 = self.attn_drop(attn1)
>                   v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C//2).
>                                           transpose(1, 2).view(B,C//2, H//self.sr_ratio, W//self.sr_ratio)).\
>                       view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2)
>                   x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C//2)
>                   attn2 = (q[:, self.num_heads // 2:] @ k2.transpose(-2, -1)) * self.scale
>                   attn2 = attn2.softmax(dim=-1)
>                   attn2 = self.attn_drop(attn2)
>                   v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C//2).
>                                           transpose(1, 2).view(B, C//2, H*2//self.sr_ratio, W*2//self.sr_ratio)).\
>                       view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2)
>                   x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C//2)
>      
>                   x = torch.cat([x1,x2], dim=-1)
>           else:
>               kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
>               k, v = kv[0], kv[1]
>               attn = (q @ k.transpose(-2, -1)) * self.scale
>               attn = attn.softmax(dim=-1)
>               attn = self.attn_drop(attn)
>               x = (attn @ v).transpose(1, 2).reshape(B, N, C) + self.local_conv(v.transpose(1, 2).reshape(B, N, C).
>                                           transpose(1, 2).view(B,C, H, W)).view(B, C, N).transpose(1, 2)
>           x = self.proj(x)
>           x = self.proj_drop(x)
>           return x
>   ~~~
> - `Detail Specofic`: 即**DW-Conv**
>
>   - ![img](https://raw.githubusercontent.com/QDDse/MD_images/main/ada64f45205645d2b0ef165d8b988451.png)
>
>   - ~~~python
>     class Mlp(nn.Module):
>         def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
>             super().__init__()
>             out_features = out_features or in_features
>             hidden_features = hidden_features or in_features
>             self.fc1 = nn.Linear(in_features, hidden_features)
>             self.dwconv = DWConv(hidden_features)
>             self.act = act_layer()
>             self.fc2 = nn.Linear(hidden_features, out_features)
>             self.drop = nn.Dropout(drop)
>          
>         def forward(self, x, H, W):
>             x = self.fc1(x)
>             x = self.act(x + self.dwconv(x, H, W))  # 残差连接，这里和图画的顺序不一样，图应该画错了
>             x = self.drop(x)
>             x = self.fc2(x)
>             x = self.drop(x)
>             return x
>          
>     class DWConv(nn.Module):
>         def __init__(self, dim=768):
>             super(DWConv, self).__init__()
>             self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
>          
>         def forward(self, x, H, W):
>             B, N, C = x.shape
>             x = x.transpose(1, 2).view(B, C, H, W)
>             x = self.dwconv(x)
>             x = x.flatten(2).transpose(1, 2)
>             return x
>     ~~~
>
>   - 





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
~~~



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
~~~

