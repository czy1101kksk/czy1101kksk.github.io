# DeTi:Training data-efficient image transformers & distillation through attention

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>

!!! info "相关信息"
    <font size = 3.5>
    
    论文地址：[Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877)

    代码（Pytorch版）:[https://github.com/facebookresearch/deit](https://github.com/facebookresearch/deit)

    </font>

### 概述

在对 ViT 的介绍中，我们了解到，ViT 算法要想取得一个较好的指标，需要先使用 JFT-300 或者 ImageNet-21K 这样的超大规模数据集进行预训练，然后再迁移到其他中等或较小规模的数据集上。而当不使用像 JFT-300 这样的巨大的数据集时，效果是不如 CNN 模型的，也就反映出 Transformer 结构在 CV 领域的一个局限性。对于大多数的研究者而言，使用如此大规模的数据集意味着需要很昂贵的计算资源，一旦无法获取到这些计算资源，不能使用这么大规模的数据集进行预训练，就无法复现出算法应有的效果。所以，出于这个动机，研究者针对 ViT 算法进行了改进，提出了DeiT。

在 DeiT 中，作者在 ViT 的基础上改进了训练策略，并使用了蒸馏学习的方式，只需要在 ImageNet 上进行训练，就可以得到一个有竞争力的 Transformer 模型，而且在单台计算机上，训练时间不到3天即可。