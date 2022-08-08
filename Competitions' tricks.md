[TOC]

# Competitions' tricks

## 1. Data Augmentation

> 数据增强： SOTA：
>
> - `cutmix`
>
> - `mixup`
>
> - [`MRA`](https://github.com/haohang96/MRA) ----利用MAE的思路进行语义增强  ,待开源
>
>   - ![img](C:\Users\int.zihao.gong\OneDrive\Notes\images\MRA)
>
>     
>
> - `GAN 生成`



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

<font size=5, color=red> 针对class unbalanced data </font>

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
>
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
>
>   - 



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
> - 

## 可视化

> `tensorboard`



#### <font size=6, color=red>WandB</font>

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

