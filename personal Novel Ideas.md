[toc]

# personal Novel Ideas

> 

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



## ViLT 后续改进

> - pretrained tasks: 
>   1. 对images引入自监督tasks（MIM） ---- **reference MAE** （FLIP）
>   1. MIM掩码的portion尝试采用adaptive  ---- 利用fc给patch打分然后筛选？（V-MoE） 
> - Attention 机制
>   1. 将原始self-attention --> 各模态内部attention + cross modal attention -- **lightweght**
>   2. 针对images 尝试多尺度训练 ---  **reference Swin-transformer**, [Gn-Conv](https://github.com/raoyongming/HorNet)
>   3. 在cross-attention中对两种模态的token之间添加bottleneck token
>   4. 在下游任务中引入meta-learning
> - Diffusion
>   1. Text-Guided style transfer  （Classifier-Guided）
> - Mapping network：
>   1. 讲CLIP的encoder 空间通过[mapping network](https://arxiv.org/abs/2111.09734)映射得到prefix，mapping network<font color=red>扩展为MoE版本</font>, 推理时<font color=red>重参数化降低latency 和 params</font> --- `Ref Clipcap, VQ-RepVGG` 
>   2. 对于上述CLIP encoder 通过mapping network映射为prefix embedding于caption进行concat时，两个模态tokens之间添加一个[bottleneck](https://arxiv.org/abs/2107.00135)进行信息fuse -- `Ref`[Code](https://github.com/google-research/scenic/tree/main/scenic/projects/mbt)
> - Distance metric
>   1. [Approximate Geodesics](https://arxiv.org/abs/2209.07496)： 近似测地线距离，“流行上的语义相似度”。-- `Ref `[Blog](https://kexue.fm/archives/9368)
>      - 常用的距离度量函数（欧式，cosine_similarity）在**流形假设下对于较远的距离不适**用，换成测地线距离可能会在某些语义场景下work
>   2. 

### 12/10-2022

> `Ideas`：
>
> - 对ViLT在VQA 微调时采用[self-supervised contrastive loss](https://arxiv.org/abs/2011.01403)
>
> ![image-20221210043041415](https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/image-20221210043041415.png)
>
> - 下载image-text retrieval 数据集 (MSCOCO, Flickr30k)
> - 用CLIP做在VQA上的微调

> `Experiments`:
>
> - [ ] 1、ViLT 利用监督对比loss 微调至VQA看看效果（huggingface ViLT？ 还是利用原始ViLT codebase）
> - [ ] 2、利用NLP Paraphrasing对text（questions）进行augmented



## ALBEF follow



## Prompt

- Contrastive loss to align the semantic space of LLMs (Forzen) and semantic space of a visible encoder.
- Using BLIP to generate or update the annotations of imgs in dataset.



## Domian Background

> vision-language model为主的多模态领域虽然目前在尺寸（model_size, data_size）还未达到NLP，或是cv的水平，但是也朝着大模型大数据方向靠拢

- NLP：GPT-3（GPT-3.5） 175B 参数量， Switch-transfomer-Trillion-level
- CV： VIT-22B
- VLM：Flamingo-9B

> 各种传感器的发展促使AI朝着更多模态的方向应用，医学领域的图像诊断，AI字幕等都是image caption的推广应用

> 大模型的预训练阶段消耗的资源是庞大的，更好的利用起已经经过大规模预训练的模型是对资源的节约。

> 微调本身对于预训练好的表征空间是一种破坏，很可能造成catastrophic forgetting

## Methods

> ` Visual Latent Prompt` ： 
>
> - 现阶段对于多模态的downstream-tasks（image-caption，VQA..etc）最后都是输出自然语言，结合目前火热的in-context learning（GPT-3），对于visual 信息固定住一个encoder（CLIP），后面利用一个mapping network（TRM OR MLP）和两个对比损失函数将视觉空间与语言空间对齐。
>
> - 将对齐后的visual embedding与对应的annotation（or QA）concat，将visual信息视为一种隐式prompt 一起经过语言模型（GPT等）输出最终结果。
>
> `Supervised Contrastive loss`：
>
> - 在表征空间对齐的过程中加入监督对比损失（SCL），利用annotation本身含有的关于对应imgs的对象表示，提取其中的某些obect作为一种更为局部标签，在同一个memory-bank中拉近其中局部标签相同的visual embedding的距离，可以视为一种intra-contrastive loss
>
> `引入近似测地线距离[Approximate Geodesics](https://arxiv.org/abs/2209.07496)代替欧式距离`： (**Ablation**)
>
> - 这种复杂的高维表征空间，本身极有可能是一种高维流形，之前用到的距离度量都是cosine-similarity（属于一种数据特征标准化后的欧氏距离），引入流形空间下的一种距离度量方式改进，叫做流形下的语言相似度。
>
> <img src="https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/image-20230220144637152.png" alt="image-20230220144637152" style="zoom:50%;" />
>
> `Bottleneck`：(**Ablation**)
>
> - 两种模态之间更加平滑的融合，可以是avg或是其他实现方式。

# 结构草图：

![image-20230220144900999](https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/image-20230220144900999.png)

------

# Related Works

## 1. VLM

| Papers                                                       | CodeBase                                                     | From                                |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------- |
| [TVLT](https://openreview.net/pdf?id=2DZ9R7GXLY)             | [Code](https://github.com/zinengtang/TVLT)                   | NeurIPS22                           |
| [GLIPv2](https://openreview.net/pdf?id=wiBEFdAvl8L)          | [GLIP](https://github.com/microsoft/GLIP)                    | NeurIPS22                           |
| [PyramidCLIP](https://openreview.net/pdf?id=7YTh6S8HIY)      | [CodeBase](https://github.com/Yuting-Gao/PyramidCLIP)        | NeurIPS22                           |
| [Visual Clues: Bridging Vision and Language Foundations for Image Paragraph Captioning](https://openreview.net/pdf?id=ZqgFbZEb8bW) |                                                              |                                     |
| [VLMo](https://openreview.net/pdf?id=bydKs84JEyw)            | [NO-Open](https://github.com/microsoft/unilm/tree/master/vlmo) | MoE of VLM, BEiT-3                  |
| [UniCLIP](https://arxiv.org/abs/2209.13430)                  |                                                              |                                     |
| [Flamingo](https://openreview.net/pdf?id=EbMuimAbPbs)        | [Pytorch](https://github.com/lucidrains/flamingo-pytorch)    |                                     |
| [LGDN: Language-Guided Denoising Network for Video-Language Modeling](https://openreview.net/pdf?id=rA2tItoRUth) |                                                              | Video-Language Denoising Network    |
| [CoT](https://openreview.net/pdf?id=_VjQlMeSB_J)             |                                                              | Prompt                              |
| [CLIPDraw: Exploring Text-to-Drawing Synthesis through Language-Image Encoders](https://openreview.net/pdf?id=c39zYHHgQmy) | [CLIPasso](https://github.com/yael-vinker/CLIPasso) [styleclipdraw](https://github.com/pschaldenbrand/styleclipdraw) | Text-Draw                           |
| [Photorealistic Text-to-Image Diffusion](https://openreview.net/pdf?id=08Yk-n5l2Al) | [imagen](https://github.com/lucidrains/imagen-pytorch)       | Text-Image Diffsion with language   |
| [[Long-Form Video-Language Pre-Training with Multimodal Temporal Contrastive Learning](https://openreview.net/forum?id=-Zzi_ZmlDiy)] | [Code](https://github.com/microsoft/XPretrain)               |                                     |
| [BMU-MoCo](https://openreview.net/pdf?id=H5z5Q--YdYd)        |                                                              |                                     |
| [Zero-Shot Video Question Answering via Frozen Bidirectional Language Models](https://openreview.net/pdf?id=9uRS5ysgb9) | [Code](https://antoyang.github.io/frozenbilm.html)           |                                     |
| [Expectation-Maximization Contrastive Learning for Compact Video-and-Language Representations](https://openreview.net/pdf?id=ijzm0EhAY_w) | [Code](https://github.com/jpthu17/EMCL)                      | Video-Language                      |
| [OmniVL](https://openreview.net/pdf?id=u4ihlSG240n)          |                                                              | Foundation Model (Video or Visual1) |
| [BinauralGrad](https://openreview.net/pdf?id=_FMJmDEPLzs)    | [Code](https://github.com/microsoft/NeuralSpeech)            | DDPM for Audio Synthesis            |
| [Mased Autoencoder that Listen](https://openreview.net/pdf?id=MAMOi89bOL) | [AudioMAE](https://github.com/facebookresearch/AudioMAE)     | MAE in Audio                        |
| [Language Models with Image Descriptors are Strong Few-Shot Video-Language Learners](https://openreview.net/pdf?id=_LceCyuVcH) | [VidIL](https://github.com/MikeWangWZHL/VidIL)               | FSL Video-Language                  |
| [Audio-Driven Co-Speech Gesture Video Generation](https://openreview.net/pdf?id=VhgC3SMTiy) | [ANGIE](https://alvinliu0.github.io/projects/ANGIE)          |                                     |
| [Multi-modal Grouping Network for Weakly-Supervised Audio-Visual Video Parsing](https://openreview.net/pdf?id=zfo2LqFEVY) | [MGN](https://github.com/stoneMo/MGN)                        |                                     |
| [Egocentric Video-Language Pretraining](https://openreview.net/pdf?id=nE8_DvxAqAB) | [EgoVLP](https://github.com/showlab/EgoVLP)                  | Video-Language pretrain             |
|                                                              |                                                              |                                     |
|                                                              |                                                              |                                     |



### 1.1 ViLT   ---- Benchmark?

> VQA

1. ViLT在VQAv2 微调时，采用的多分类的形式，属于**Close-ended setting**（另一种是**open-ended setting**）
2. 

### 1.2 ALBEF

本文总结了CLIP，Align等双塔模型和ViLT, OSCAR等单塔模型的优势，提出：**（SalesForce Research）**

1. Vision encoder理论上要比Text encoder更复杂
2. 直接对多模态的tokens做attention fuse的交互可能比较challenging
3. 多模态本身开源的dataset量级并不算大，广泛使用网络爬取到的image-text pairs dataset(CC, SUB)是带有大量noisy的。

<img src="https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/image-20221218154209279.png" alt="image-20221218154209279" style="zoom:67%;" />

**本文贡献：**

1. 提出如上所示的模型结构，采用12层标准**ViT-B/16(85.8M**)对images(256X256)进行encoding，同时使用![img](http://wiki.enflame.cn/plugins/servlet/latexmath/placeholder?key=0eb5b6b910b174a27f21af2a29699e35&vertAlign=-5px)12层(123.7M)中的前6层对text进行encoding，满足了visual encoder > text encoder的假设；

2. 利用CLIP的对比学习方式先进行了align（ITC），之后concat之后经过BERT后6层encoder进行模态交互，采用MLM、ITM两种pretrained tasks，采用herd negative挖掘与pos sample最相似的负样本增加pretrained tasks的难度。

3. 借鉴MoCo采用<font color=red>**Momentum Model**</font>对noise的image-text dataset进行优化，具体做法就是基于**EMA的self-training**（or pseudo labels from older model）。

4. 作者pretrained dataset延续了UNITER在两种web dataset（Conceptual Captions and SBU Captions）和两种In-domain dataset（COCO and Visual Genome）；在各种downstream tasks都取得不错的效果（VQA，NLVR， VG， ITR，TR）:
   <img src="https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/image-20221218154256564.png" alt="image-20221218154256564" style="zoom:67%;" />

5. Evaluation

   <img src="https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/image-20221218154328734.png" alt="image-20221218154328734" style="zoom:50%;" />

### 1.3 Plug-and-Play VQA (Salesforce Reserch) --- 22/11 arXiv

[Plug-and-Play VQA: Zero-shot VQA by Conjoining Large Pretrained Models with Zero Training](https://arxiv.org/abs/2210.08773)

[CodeBase](https://github.com/salesforce/LAVIS/tree/main/projects/pnp-vqa#plug-and-play-vqa-zero-shot-vqa-by-conjoining-large-pretrained-models-with-zero-training)

<img src="https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/image-20221217180342867.png" alt="image-20221217180342867" style="zoom: 50%;" />

<center style="color:#C0C0C0;text-decoration:underline">OverView of PNP-VQA</center>

> `Method`:
>
> - 作为一种plug and play modular architecture，将三个modules 进行合并：
>   - **Image-Question Matching**： 对input（img-question pairs）判断二分类（match/dismatch）
>   - **Image Caption**: 对上一步pairs中img划分patch，同时根据与question最相似原则（**relevancy score**）进行patch 采样得到k-patch，将k-patch经过image caption module生成**question-guided N-caption**
>   - **Question-Answering Module**：将N-captions and question输入QA-Module输出answer

> `Challenges`:

1. 现有的模型大都需要针对下游任务做出微调，这些工作需要对预训练PLMs进行大量的适应。

> `Related Work`:

1. 大规模img-text pretrained
2. **自然语言作为一种中间表示**： 
   - 自然语言作为不同模型或是多个推理步骤之间的intermediate presentation 是一种新兴的ML strategies(Learning with Latent Language-- 引用不到一百)

> `Contributions`:

1. 提出一种modular framework，即插即用，可以随着预训练模型的进步实现共同进步
2. 通过可解释性技术，创建广泛覆盖于Q相关信息的image captions，进而实现准确的QA
3. 在两个dataset上（VQAv2，GQA）实现zero-shot accuracy 的sota（63.3）



### 1.4 BLIP (Salesforce Reserch) --ICML2022

[BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086)

[CodeBase](https://github.com/salesforce/BLIP)

![BLIP.gif](https://github.com/salesforce/BLIP/blob/main/BLIP.gif?raw=true)

>  Question:

1. Most existing methods只是在understanding-based tasks 或者generation-based tasks各自实现进步
2. Visual-text dataset 目前没有做到large-scale并且来自web的data本身的标注十分noisy

> Contributions:

1. **MED**(Multi-modal mixture of encoder-decoder): **Text-encoder** and **Image-grounded Text encoder**  and **Image-grounded Text decoder** 三个module组成

2. **CapFilt**（Captioner + Filter）对数据imgs经过captioner生成synthetic annotations 并通过filter筛选，可以看作一种self-training的变体

   <img src="https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/image-20221218154822557.png" alt="image-20221218154822557" style="zoom:67%;" />

   <center style="color:#C0C0C0;text-decoration:underline">OverView of CapFilt</center>

   3. 预训练参考了自己团队中ALBEF的hard negative，pytorch在32卡训练



### 1.5 CoCoOp -- CVPR22  

> `Conditional Context Optimization`

[Conditional Prompt Learning for Vision-Language Models](https://arxiv.org/abs/2203.05557)

[CodeBase](https://github.com/KaiyangZhou/CoOp)

> `Challenges`

1. CoOp(Context Optimization) 无法对同一dataset中的其他unseen class做出好的判断，泛化性不行
2. 利用自然语言所含有的丰富语义概念代替label categories当作监督信号具有广阔的应用前景（CLIP，Align, etc...），可以实现由close-set visual concept 到 open-set visual concept
3. 对于在大规模数据集上进行训练的模型，如果finetune 在下游任务，会对已经学习到的<font color=red>表征空间造成损害</font>

<img src="https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/image-20221218160855205.png" alt="image-20221218160855205" style="zoom:67%;" />

<center style="color:#C0C0C0;text-decoration:underline">OverView of CoCoOp</center>

> `Contributions`

1. 在本组之前的工作过[CoOp]()进行改进，提高了unseen labels 的泛化性
2. `Learnable module：`

   1. Meta-net: 将visual tokens through Image Encoder 通过一个轻量级的MLP（FC-ReLU-FC）映射为所谓的meta tokens

   2. Context tokens：该概念来自于CoOp，M leanrnable vectors: 
      $$
      t_i = \{v_1, v_2, ..., v_M, c_i\}
      $$
      where v_i 有着相同的word embedding dimension

3. 参数量对比：

<img src="https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/image-20221217180014232.png" alt="image-20221217180014232" style="zoom: 50%;" />

4. 结果表示在unseeb class 上表现效果更好，但是在Base class 上表现效果一般般，甚至不如CoOp原始，并且训练参数也比原始CoOp要高。

### 1.6 PyramidCLIP -- NeurIPS22

[paper](https://arxiv.org/abs/2204.14095)

[CodeBase](https://github.com/Yuting-Gao/PyramidCLIP)

<img src="https://raw.githubusercontent.com/QDDse/MD_images/main/MD_images/PyramidCLIP-figure.jpg" alt="image" style="zoom: 80%;" />



### 1.7 BLIP

### 1.8 img2prompt



## 2. Dataset

> `Pretrained Dataset`

|    Datasets    |                             Link                             | desription                                                   |
| :------------: | :----------------------------------------------------------: | ------------------------------------------------------------ |
| **LAION-400M** | [link_parquet](https://www.kaggle.com/datasets/romainbeaumont/laion400m ) | ~50G                                                         |
|    **GCC**     | [ConceptualCaptions](https://ai.google.com/research/ConceptualCaptions/download) | https://github.com/salesforce/LAVIS/blob/main/dataset_card/conceptual_captions.md |
|    **SBU**     | [sbucaptions](http://www.cs.virginia.edu/~vicente/sbucaptions/) | https://github.com/salesforce/LAVIS/blob/main/dataset_card/sbu_caption.md |
|     **VG**     | [VisualGenome](http://visualgenome.org/api/v0/api_home.html) | Download [image part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [image part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip) and [region descriptions](http://visualgenome.org/static/data/dataset/region_descriptions.json.zip) |
|    **COCO**    |          [COCO](https://cocodataset.org/#download)           | https://github.com/salesforce/LAVIS/blob/main/dataset_card/coco_caption.md |

> `Downstream Tasks Dataset`

|    Datasets    |                    Link                     | Description                                                  |
| :------------: | :-----------------------------------------: | ------------------------------------------------------------ |
|   **VQAv2**    | [VQAv2](https://visualqa.org/download.html) | Download COCO [2014 train images](http://images.cocodataset.org/zips/train2014.zip), [2014 val images](http://images.cocodataset.org/zips/val2014.zip), [2015 test images](http://images.cocodataset.org/zips/test2015.zip), annotations ([train](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip), [val](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip)), and questions ([train](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip), [val](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip), [test](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip)) |
| **NLVR2/NLVR** |   [nlvr](https://github.com/lil-lab/nlvr)   |                                                              |
|                |                                             |                                                              |



### 2.1 KILOGRAM -- EMNLP best paper 22

[Codebase](https://github.com/lil-lab/kilogram)

- 整体数据是一种七巧板式的抽象images-text，审稿人的品味很special。。
- 目前不知道后续是否有足够多的works 跟进， 观望。。

### 

## 3. Augmentation

### 3.1 NLP paraphrasing

> `Reference`:
>
> - [Paraphrasing in Natural Language Processing (NLP)](https://lopezyse.medium.com/paraphrasing-in-natural-language-processing-nlp-857c28e68488)

> `Intro`:
>
> - Paraphrasing in NLP： Language generation task, 输出的句子保留有输入的语义，但是在单词选择和语言有所变化

> `Application`:
>
> - Huggingface： https://colab.research.google.com/drive/1RWvGuHKnPur7fCL0DObMeZXQVHem6aEV?usp=sharing#scrollTo=8IeJIWrzoTHB
>
>   ~~~python
>   !pip install --upgrade transformers sentencepiece
>         
>   import torch
>   from transformers import PegasusForConditionalGeneration, PegasusTokenizer
>   model_name = 'tuner007/pegasus_paraphrase'
>   torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
>   tokenizer = PegasusTokenizer.from_pretrained(model_name)
>   model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
>         
>   ## Paraphrasing Function Using PEGASUS
>   def get_response(input_text,num_return_sequences,num_beams):
>     batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
>     translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
>     tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
>     return tgt_text
>   ## demo
>   context = "The ultimate test of your knowledge is your capacity to convey it to another."
>   num_return_sequences=10
>   num_beams=10
>   get_response(context,num_return_sequences,num_beams)
>         
>   ## demo on questions
>   context = "Which course should I take to get started in data science?"
>   num_return_sequences=10
>   num_beams=10
>   get_response(context,num_return_sequences,num_beams)
>   '''
>   ['Which data science course should I take?',
>    'Which data science course should I take first?',
>    'Should I take a data science course?',
>    'Which data science class should I take?',
>    'Which data science course should I attend?',
>    'I want to get started in data science.',
>    'Which data science course should I enroll in?',
>    'Which data science course is right for me?',
>    'Which data science course is best for me?',
>    'Which course should I take to get started?']
>    '''
>   ~~~
>
>   

