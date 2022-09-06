[TOC]

# Competitions' tricks

## 1. Data Augmentation

> 数据增强： SOTA：
>
> - **`mixup & Cutmix`**
>
>   - ~~~python
>     ## 手动实现
>     def rand_bbox(size, lam):
>         W = size[2]
>         H = size[3]
>         cut_rat = np.sqrt(1. - lam)
>         cut_w = np.int(W * cut_rat)
>         cut_h = np.int(H * cut_rat)
>       
>         # uniform
>         cx = np.random.randint(W)
>         cy = np.random.randint(H)
>       
>         bbx1 = np.clip(cx - cut_w // 2, 0, W)
>         bby1 = np.clip(cy - cut_h // 2, 0, H)
>         bbx2 = np.clip(cx + cut_w // 2, 0, W)
>         bby2 = np.clip(cy + cut_h // 2, 0, H)
>       
>         return bbx1, bby1, bbx2, bby2
>     def cutmix(data, targets1, targets2, targets3, alpha):
>         indices = torch.randperm(data.size(0))
>         shuffled_data = data[indices]
>         shuffled_targets1 = targets1[indices]
>         shuffled_targets2 = targets2[indices]
>         shuffled_targets3 = targets3[indices]
>       
>         lam = np.random.beta(alpha, alpha)
>         bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
>         data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
>         # adjust lambda to exactly match pixel ratio
>         lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
>       
>         targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]
>         return data, targets
>       
>     def mixup(data, targets1, targets2, targets3, alpha):
>         indices = torch.randperm(data.size(0))
>         shuffled_data = data[indices]
>         shuffled_targets1 = targets1[indices]
>         shuffled_targets2 = targets2[indices]
>         shuffled_targets3 = targets3[indices]
>       
>         lam = np.random.beta(alpha, alpha)
>         data = data * lam + shuffled_data * (1 - lam)
>         targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]
>       
>         return data, targets
>       
>     ~~~
>     
>   - ~~~py
>     ## 利用torchtoolbox以及Timm实现!!
>     !pip install torchtoolbox
>     from torchtoolbox.transform import Cutout
>       
>     ### 数据预处理
>     transform = transfroms.Compose([
>         transforms.Resize((224, 224)),
>         Cutout(),
>         transforms.ToTensor(),
>         transform.Normlize([0.5,0.5,0.5],[0.5,0.5,0.5]),
>     ])
>     ### 导入timm
>     from timm.data.mixup import Mixup
>     from timm.loss import SoftTargetCrossEntropy
>     #### 定义mixup_fn
>     mixup_args = {
>         'mixup_alpha': 1.,
>         'cutmix_alpha': 0.,
>         'cutmix_minmax': None,
>         'prob': 1.0,
>         'switch_prob': 0.,
>         'mode': 'batch',
>         'label_smoothing': 0,
>         'num_classes': 1000}
>     mixup_fn = 
>     criterion_train = SoftTargetCrossEntropy()
>           
>     ~~~
>
>   - 
>
> - [`MRA`](https://github.com/haohang96/MRA) ----利用MAE的思路进行语义增强  ,待开源
>
>   - ![img](C:\Users\int.zihao.gong\OneDrive\Notes\images\MRA)
>
> - `GAN 生成`

### 1.1 SAM

[paper](https://arxiv.org/abs/2010.01412)

[github](https://github.com/davda54/sam)

> `motivation`：
>
> - 由于现代DNN网络参数量巨大，导致在空间中存在众多极限值点：
>   - `sharp minimal`:  对input和参数更敏感
>   - `flat minimal`：具有更高的generalization
>   - ![preview](https://pic2.zhimg.com/v2-3ac8250b0cba2921d6b2d1ceb736d65d_r.jpg)

~~~py
from sam import SAM
...

model = YourModel()
base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
optimizer = SAM(model.parameters(), base_optimizer, lr=0.1, momentum=0.9)
...

for input, output in data:

  # first forward-backward pass
  loss = loss_function(output, model(input))  # use this loss for any training statistics
  loss.backward()
  optimizer.first_step(zero_grad=True)
  
  # second forward-backward pass
  loss_function(output, model(input)).backward()  # make sure to do a full forward pass
  optimizer.second_step(zero_grad=True)
...
~~~



### some kits

> - `scikit-image` 
> - `FightingCV`: 用于学习以及直接调用的[repo](https://github.com/xmu-xiaoma666/FightingCV-Paper-Reading)



> + `einsum`: Einstein summation convention---爱因斯坦求和约定
>
>   + 矩阵求迹
>   + 求矩阵对角线 diga
>   + Tensor （along axis）求和： sum
>   + transpose
>   + element-wise product: 哈达玛积
>   + 矩阵点乘：dot
>   + Tensor 乘法：tensordot
>   + 向量内积：inner
>   + 外积：outer
>
> + ~~~python
>   ## 一套用法
>   einsum(equation, *operands)
>                                     
>   1
>   ~~~
>
> + 



## 2 .Loss

### 2.1 <font size=5, color=red> 针对class unbalanced data（Focal loss） </font>

[参考Repo](https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/class_balanced_loss.py)

> - `Focal Loss`:
>
>   - ~~~python
>     import numpy as np
>     import torch
>     import torch.nn.functional as F
>
> 
>
<<<<<<< HEAD
>     def focal_loss(labels, logits, alpha, gamma):
>         """Compute the focal loss between `logits` and the ground truth `labels`.
>         Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
>         where pt is the probability of being classified to the true class.
>         pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
>         Args:
>           labels: A float tensor of size [batch, num_classes].
>           logits: A float tensor of size [batch, num_classes].
>           alpha: A float tensor of size [batch_size]
>             specifying per-example weight for balanced cross entropy.
>           gamma: A float scalar modulating loss from hard and easy examples.
>         Returns:
>           focal_loss: A float32 scalar representing normalized total loss.
>         """    
>         BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")
>                         
>         if gamma == 0.0:
>             modulator = 1.0
>         else:
>             modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
>                 torch.exp(-1.0 * logits)))
>                                         
>         loss = modulator * BCLoss
>                                         
>         weighted_loss = alpha * loss
>         focal_loss = torch.sum(weighted_loss)
>                                         
>         focal_loss /= torch.sum(labels)
>         return 
>     ~~~
=======
> ```python
> def focal_loss(labels, logits, alpha, gamma):
>     """Compute the focal loss between `logits` and the ground truth `labels`.
>     Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
>     where pt is the probability of being classified to the true class.
>     pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
>     Args:
>       labels: A float tensor of size [batch, num_classes].
>       logits: A float tensor of size [batch, num_classes].
>       alpha: A float tensor of size [batch_size]
>         specifying per-example weight for balanced cross entropy.
>       gamma: A float scalar modulating loss from hard and easy examples.
>     Returns:
>       focal_loss: A float32 scalar representing normalized total loss.
>     """    
>     BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")
> 
>     if gamma == 0.0:
>         modulator = 1.0
>     else:
>         modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
>             torch.exp(-1.0 * logits)))
> 
>     loss = modulator * BCLoss
> 
>     weighted_loss = alpha * loss
>     focal_loss = torch.sum(weighted_loss)
> 
>     focal_loss /= torch.sum(labels)
>     return 
> 
> ```
### 2.2 Lbael Smooth 
> - `CB_Loss`
>
>   - ~~~python
>     def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
>         """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
>         Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
>         where Loss is one of the standard losses used for Neural Networks.
>         Args:
>           labels: A int tensor of size [batch].
>           logits: A float tensor of size [batch, no_of_classes].
>           samples_per_cls: A python list of size [no_of_classes].
>           no_of_classes: total number of classes. int
>           loss_type: string. One of "sigmoid", "focal", "softmax".
>           beta: float. Hyperparameter for Class balanced loss.
>           gamma: float. Hyperparameter for Focal loss.
>         Returns:
>           cb_loss: A float tensor representing class balanced loss
>         """
>         effective_num = 1.0 - np.power(beta, samples_per_cls)
>         weights = (1.0 - beta) / np.array(effective_num)
>         weights = weights / np.sum(weights) * no_of_classes
>                                         
>         labels_one_hot = F.one_hot(labels, no_of_classes).float()
>                                         
>         weights = torch.tensor(weights).float()
>         weights = weights.unsqueeze(0)
>         weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
>         weights = weights.sum(1)
>         weights = weights.unsqueeze(1)
>         weights = weights.repeat(1,no_of_classes)
>                                         
>         if loss_type == "focal":
>             cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
>         elif loss_type == "sigmoid":
>             cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
>         elif loss_type == "softmax":
>             pred = logits.softmax(dim = 1)
>             cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
>         return cb_loss
>                                         
>     ~~~

- `Label-Smoothing`:
>
>  - ~~~python
>    class LabelSmoothing(nn.Module):
>        """NLL loss with label smoothing.
>                        """
>        def __init__(self, smoothing=0.0):
>            """Constructor for the LabelSmoothing module.
>            :param smoothing: label smoothing factor
>            """
>        	                    super(LabelSmoothing, self).__init__()
>            self.confidence = 1.0 - smoothing
>            self.smoothing = smoothing
>            # 此处的self.smoothing即我们的epsilon平滑参数。

>                    def forward(self, x, target):
>            logprobs = torch.nn.functional.log_softmax(x, dim=-1)
>            nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
>            nll_loss = nll_loss.squeeze(1)
>            smooth_loss = -logprobs.mean(dim=-1)
>                        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
>            return loss.mean()




## 3. 参考model

### 3.1 CLIP --- Contrastive Language-Image Pretrained

> - `Openai`:
>
>   - **Installation and cite**
>
>     ~~~python
>     !pip install git+https://github.com/openai/CLIP.git
>     
>     import clip
>     ## 查看可用model
>     clip.available_models()
>     model, preprocess = clip.load("ViT-B/32")  # model, preprocess 是一个torchvision.transfrom.Compose
>     model.cuda().eval()
>     input_resolution = model.visual.input_resolution
>     context_length = model.context_length
>     vocab_size = model.vocab_size
>     ##Text Tokenize  ## 默认padding 至77tokens
>     clip.tokenize("Hello World!")
>     
>     ##images in skimage to use and their textual descriptions
>     descriptions = {
>         "page": "a page of text about segmentation",
>         "chelsea": "a facial photo of a tabby cat",
>         "astronaut": "a portrait of an astronaut with the American flag",
>         "rocket": "a rocket standing on a launchpad",
>         "motorcycle_right": "a red motorcycle standing in a garage",
>         "camera": "a person looking at a camera on a tripod",
>         "horse": "a black-and-white silhouette of a horse", 
>         "coffee": "a cup of coffee on a saucer"
>     }
>     ## image-Text 经过各自的encoder
>     image_input = torch.tensor(np.stack(images)).cuda()
>     text_tokens = clip.tokenize(["This is " + desc for desc in texts]).cuda()
>     ## 计算cosine similarity
>     image_features /= image_features.norm(dim=-1, keepdim=True)
>     text_features /= text_features.norm(dim=-1, keepdim=True)
>     similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T  #简单的矩阵乘法
>     ## 可视化heatmap
>     count = len(descriptions)
>     plt.figure(figsize=(20, 14))
>     plt.imshow(similarity, vmin=0.1, vmax=0.3)
>     # plt.colorbar()
>     plt.yticks(range(count), texts, fontsize=18)
>     plt.xticks([])
>     for i, image in enumerate(original_images):
>         plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
>     for x in range(similarity.shape[1]):
>         for y in range(similarity.shape[0]):
>             plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)
>     
>     for side in ["left", "top", "right", "bottom"]:
>       plt.gca().spines[side].set_visible(False)
>     
>     plt.xlim([-0.5, count - 0.5])
>     plt.ylim([count + 0.5, -2])
>     
>     plt.title("Cosine similarity between text and image features", size=20)
>     ~~~
>
>   - `Zero-shot classification`
>
>     - ~~~python
>       text_descriptions = [f"This is a photo of a {label}" for label in cifar100.classes]
>       text_tokens = clip.tokenize(text_descriptions).cuda()
>                                                                                                       
>       with torch.no_grad():
>           text_features = model.encode_text(text_tokens).float()
>           text_features /= text_features.norm(dim=-1, keepdim=True)
>                                                                                                       
>       text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
>       top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)
>       # 可视化
>       plt.figure(figsize=(16, 16))
>                                                                                                       
>       for i, image in enumerate(original_images):
>           plt.subplot(4, 4, 2 * i + 1)
>           plt.imshow(image)
>           plt.axis("off")
>                                                                                                       
>           plt.subplot(4, 4, 2 * i + 2)
>           y = np.arange(top_probs.shape[-1])
>           plt.grid()
>           plt.barh(y, top_probs[i])
>           plt.gca().invert_yaxis()
>           plt.gca().set_axisbelow(True)
>           plt.yticks(y, [cifar100.classes[index] for index in top_labels[i].numpy()])
>           plt.xlabel("probability")
>                                                                                                       
>       plt.subplots_adjust(wspace=0.5)
>       plt.show()
>       ~~~



## 4. 多尺度训练

> [FixRes](https://arxiv.org/abs/1906.06423)
>
> - 用`original resolution`训练以后用更大的resolution进行finetune



## 5. 模型参数

### 5.1 EMA(指数移动平均)

> - `EMA（指数移动平均）`:  
>
>   - ~~~python
>     ## EMA 的Pytorch实现
>     ### From https://zhuanlan.zhihu.com/p/68748778
>     class EMA():
>         def __init__(self, model, decay):
>             '''
>             model: 训练的模型
>             decay: β 值 （vt = β*vt-1 + （1-β）*θt）
>             '''
>             self.model = model
>             self.decay = decay
>             self.shadow = {}
>             self.backup = {}
>                                                                     
>     	def register(self):
>             for name, param in self.model.named_parameters():
>             	if param.required_grad:
>                     self.shadow[name] = param.data.clone()
>                                                                             
>         def update(self):
>             for name, param in self.model.named_parameters():
>                 if param.required_gard:
>                 	assert name in self.shadow
>                     new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
>                     self.shadow[name] = new_average.clone()
>                                                                             
>         def apply_shadow(self):
>             for name, param in self.model.named_parameters():
>                 if param.required_grad():
>                     assert name in self.shadow
>                     self.backup[name] = param.data
>                     param.data = self.shadow[name]
>         def restore(self):
>             for name, param in self.model.named_parameters():
>                 if param.required_grad:
>                     assert name in self.backup
>                     param.data = self.backup[name]
>             self.backup = {}
>                                                                     
>     ## 初始化EMA
>     ema = EMA(model, decay=0.9)
>     ema.register()
>     # 训练过程中，更新完参数后，同步update shadow weights
>     def train():
>         optimizer.step()
>         ema.update()
>                                                             
>     # eval前，apply shadow weights；eval之后，恢复原来模型的参数
>     def evaluate():
>         ema.apply_shadow()
>         # evaluate
>         ema.restore()
>                                                                             
>     ~~~

### 5.2 Early-Stop

~~~ python
## 手动进行早停 （es == patience）
if val_acc > best_acc:
    best_acc = val_acc
    es = 0
    torch.save(net.state_dict(), "model_" + str(fold) + 'weight.pt')
else:
    es += 1
    print("Counter {} of 5".format(es))

    if es > 4:
        print("Early stopping with best_acc: ", best_acc, "and val_acc for this epoch: ", val_acc, "...")
        break
~~~

##  <font size=6, color=red>6. WandB</font>

### 6.1 可视化

> `tensorboard`

> - `step1`: login： 
>
>   - ~~~bash
>     wandb login
>     ### 输入api key
>     ~~~
>
> - `step2`: wandb.init
>
>   - ~~~python
>     import wandb
>     run = wandb.init(
>         project='xxx', 
>         entity='qddse',
>         name='Effi-b6-baseline',
>         reinit=True,
>         config=opt, # argparse的namespace
>         group='xxx' # 将不同的run分为一组，方便对比
>     )
>     ~~~
>
> - `step3`: run.log()
>
>   - ~~~python
>     run.log({'epoch:': epoch, 'loss_train:': epoch_l})
>     run.log({'epoch:': epoch, 'time_train:': epoch_t})
>     ~~~
>
> - `step4`: if need
>
>   - ~~~bash
>     ## 有时候本地run file 没有同步到云端，需要手动sync
>     wandb sync ./wandb/latest-run
>     ~~~
>
>     

### 6.2 炼丹 --（调参）

> `Sweep`: 可用来网格搜索， random 搜索， 贝叶斯优化
>
> `Demo`:
>
> ~~~python
> !pip install wandb -Uq
> import wandb
> import torch
> import torch.optim as optim
> import torch.nn.functional as F
> import torch.nn as nn
> from torchvision import datasets, transforms
> 
> wandb.login()
> 
> ## define sweep
> sweep_config = {
>     'method': 'random' ## [grid, random, bayes]
> }
> metric = {
>     'name': 'loss',
>     'goal': 'minimize'
> }
> sweep_config['metric'] = metric
> ## params
> parameters_dict = {
>     'optimizer': {
>         'values': ['adam', 'sgd']
>     },
>     'fc_layer_size':{
>         'values': [128, 256, 512]
>     },
>     'dropout': {
>         'values': [0.3, 0.4, 0.5]
>     },
> }
> sweep_config['parameters'] = parameters_dict
> ## 更新
> parameters_dict.update(
>     {
>         'epochs': {
>             'value': 1
>         }
>     }
> )
> parameters_dict.update({
>     'learning_rate': {
>         # a flat distribution between 0 and 0.1
>         'distribution': 'uniform',
>         'min': 0,
>         'max': 0.1
>       },
>     'batch_size': {
>         # integers between 32 and 256
>         # with evenly-distributed logarithms 
>         'distribution': 'q_log_uniform_values',
>         'q': 8,
>         'min': 32,
>         'max': 256,
>       }
>     })
> ## pprint sweep_config
> import pprint
> pprint.pprint(sweep_config)
> 
> ## define sweep and get the sweep_id
> sweep_id = wandb.sweep(sweep_config, project='xxx')
> ##
> device = torch.device('cpu')
> if torch.cuda.is_available():
>     device = torch.device('cuda:0')
>     if torch.cuda.device_count() > 1:
>         net = torch.
> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
> 
> def train(config=None):
>     # Initialize a new wandb run
>     with wandb.init(config=config):
>         # If called by wandb.agent, as below,
>         # this config will be set by Sweep Controller
>         config = wandb.config
> 
>         loader = build_dataset(config.batch_size)
>         network = build_network(config.fc_layer_size, config.dropout)
>         optimizer = build_optimizer(network, config.optimizer, config.learning_rate)
> 
>         for epoch in range(config.epochs):
>             avg_loss = train_epoch(network, loader, optimizer)
>             wandb.log({"loss": avg_loss, "epoch": epoch})
>             
> def build_dataset(batch_size):
>    
>     transform = transforms.Compose(
>         [transforms.ToTensor(),
>          transforms.Normalize((0.1307,), (0.3081,))])
>     # download MNIST training dataset
>     dataset = datasets.MNIST(".", train=True, download=True,
>                              transform=transform)
>     sub_dataset = torch.utils.data.Subset(
>         dataset, indices=range(0, len(dataset), 5))
>     loader = torch.utils.data.DataLoader(sub_dataset, batch_size=batch_size)
> 
>     return loader
> 
> ## define some functions
> def build_network(fc_layer_size, dropout):
>     network = nn.Sequential(  # fully-connected, single hidden layer
>         nn.Flatten(),
>         nn.Linear(784, fc_layer_size), nn.ReLU(),
>         nn.Dropout(dropout),
>         nn.Linear(fc_layer_size, 10),
>         nn.LogSoftmax(dim=1))
> 
>     return network.to(device)
>         
> 
> def build_optimizer(network, optimizer, learning_rate):
>     if optimizer == "sgd":
>         optimizer = optim.SGD(network.parameters(),
>                               lr=learning_rate, momentum=0.9)
>     elif optimizer == "adam":
>         optimizer = optim.Adam(network.parameters(),
>                                lr=learning_rate)
>     return optimizer
> 
> 
> def train_epoch(network, loader, optimizer):
>     cumu_loss = 0
>     for _, (data, target) in enumerate(loader):
>         data, target = data.to(device), target.to(device)
>         optimizer.zero_grad()
> 
>         # ➡ Forward pass
>         loss = F.nll_loss(network(data), target)
>         cumu_loss += loss.item()
> 
>         # ⬅ Backward pass + weight update
>         loss.backward()
>         optimizer.step()
> 
>         wandb.log({"batch loss": loss.item()})
> 
>     return cumu_loss / len(loader)
> ## run the sweepp
> wandb.agent(sweep_id, train, count=5) ##count: trial_num
> ~~~



## 数据分箱--（Split-Bin）

> `Function`:
>
> - 提高模型的稳定性和鲁棒性
> - 防止Overfitting
> - Accelerate Trainig
> - 很多的处理空值和缺失值
> - 增强逻辑回归的拟合力
> - ![img](https://img-blog.csdnimg.cn/20210507111752619.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L09yYW5nZV9TcG90dHlfQ2F0,size_16,color_FFFFFF,t_70)

~~~python
## 实现
### 无监督分箱
import pandas as pd
df = pd.DataFrame({'Age': [29,7,49,12,50,34,36,75,61,20,3,11]})

df['等距'] = pd.cut(df['Age'], bins=4)
df['等频'] = pd.qcut(df['Age'], 4)

### 有监督分箱  ## 本身pandas 版本不能太高
import scorecardpy as sc
df = pd.DataFrame({'年龄': Age,
                   'Y'   : Y})

bins = sc.woebin(df, y='Y', method='tree')  # 决策树分箱
sc.woebin_plot(bins)

~~~

## EDA

### 1. Heatmap

> - `Seaborn`: 
>
>   - ~~~py
>     ## sns, pandas
>     import seaborn as sns
>     import pandas as pd
>     colormap = sns.color_palette('Blues')
>     sns.heatmap(train.corr(), annot=True, cmap=colormap)
>     
>     ~~~
>
>   ![image-20220907012035883](https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/image-20220907012035883.png)







# Pytorch Tricks

## 1. Gumbel-Softmax Trick

[`知乎`](<img src="https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/image-20220830145134844.png"/>)

> 从离散分布中采样不能求导BP， 因此设计一个公式用于求导
>
> ![image-20220830145030706](https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/image-20220830145030706.png)
>
> 用`Softmax` 代替不可导的`Argmax`
>
> ![image-20220830145134844](https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/image-20220830145134844.png)

~~~ python
## Pytorch 实现
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    
    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard
~~~

### 2. 指定GPU编号

> 设置当前使用的GPU设备 为0号：
>
> - `ps.environ['CUDA_VISIBLE_DEVICES'] = 0`
>
> 设置当前使用的GPU设备为0,1
>
> - `os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'`
> - 根据顺序优先使用gpu0

### 3. 查看模型每层输出情况

~~~py
## Pytorch实现
from torchinfo import summary

~~~

### 4. 梯度裁剪（Gradient Clipping）

> `max_norm`: 梯度的最大范数
>
> `norm_type`: 规定范数的类型， 默认为L2

~~~py
import torch.nn as nn

outputs = model(input)
loss = loss_fn(outputs, target)
optimizer.zero_grad()
loss.backward()
nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
optimizer.step()

~~~

### 5. 冻结某些层参数

~~~py
## 
net = Network()
for name, value in  net.named_parameters():
    print('name: {0}, \t  grad: {1}'.format(name, value.required_grad))
    
## 定义冻结的层
no_grad = [
    'cnn.VGG_16.convolution1_1.weight',
    'cnn.VGG_16.convolution1_1.bias',
    'cnn.VGG_16.convolution1_2.weight',
    'cnn.VGG_16.convolution1_2.bias'
]
net = Net.CTPN()  # 获取网络结构
for name, value in net.named_parameters():
    if name in no_grad:
        value.requires_grad = False
    else:
        value.requires_grad = True
~~~

### 6. 对不同的层使用不同的lr

~~~py

net = Network()  # 获取自定义网络结构
for name, value in net.named_parameters():
    print('name: {}'.format(name))

# 输出：
# name: cnn.VGG_16.convolution1_1.weight
# name: cnn.VGG_16.convolution1_1.bias
# name: cnn.VGG_16.convolution1_2.weight
# name: cnn.VGG_16.convolution1_2.bias
# name: cnn.VGG_16.convolution2_1.weight
# name: cnn.VGG_16.convolution2_1.bias
# name: cnn.VGG_16.convolution2_2.weight
# name: cnn.VGG_16.convolution2_2.bias

conv1_params = []
conv2_params = []

for name, parms in net.named_parameters():
    if "convolution1" in name:
        conv1_params += [parms]
    else:
        conv2_params += [parms]

# 然后在优化器中进行如下操作：
optimizer = optim.Adam(
    [
        {"params": conv1_params, 'lr': 0.01},
        {"params": conv2_params, 'lr': 0.001},
    ],
    weight_decay=1e-3,
)
~~~

