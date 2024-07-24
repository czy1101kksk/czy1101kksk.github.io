# 🛣Stanford CS231n:Deep Learning for Computer Vision  
---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>

!!! info "想说的话🎇"
    
    <font size = 3.5>
    
    🔝课程网站：[https://cs231n.stanford.edu/](https://cs231n.stanford.edu/)
    
    2024版PPT: [https://cs231n.stanford.edu/slides/2024/](https://cs231n.stanford.edu/slides/2024/)
    
    </font>

###  Sequence to Sequence with RNNs 
---

![](./cs231-img/seq2seq.png)

左侧的RNN（Encoder:$h_t=f_w(x_t,h_{t-1})$）将输入序列编码总结成2个向量（$s_0$，$c$），$s_0$作为解码器的初始状态(initial decoder state,或者设置为0)，$c$作为解码器的上下文向量（Context vector，tansfer encoded sequence information to the decoder）。

右侧的RNN(decoder)将这个向量解码成输出序列。

- During Training:
    在训练网络过程中，每次不使用上一个state的输出作为下一个state的输入，而是直接使用训练数据的标准答案(ground truth)的对应上一项作为下一个state的输入,不管输出是否正确（teacher-forcing）

    ![](./cs231-img/tea.png)

- During Test-time:

    我们从输出中进行抽样，直到抽中```[STOP]```

显然，这个Seq2seq模型并不适用于长文本任务
，因为如果输入序列过长，基于RNN的编码器没有能力去捕捉足够的信息，导致解码器无法生成准确的输出。并且，希望用单一的上下文向量$c$去总结整个长序列信息，显然是不现实的。

我们可以想象一种算法，不是使用单个的上下文向量$c$，
而是在decoder的每个时间步中计算一个上下文向量，即给予decoder专注于输入序列的不同部分，选择或者重建一个新的上下文向量的能力。

![](./cs231-img/Bahdanau.png)

如上，我们编写一个对齐函数$f_{att}$（alignment function，通常为MLPs），将Encoder的隐藏状态与$s_0$输入得到alignment scores（how much should we attend to each hidden state of encoder），然后使用softmax函数归一化得到权重$a_{t,i}$（attention weights）。

得到权重后，我们使用加权求和得到上下文向量$c_t$，即：
$$
c_t=\sum_{i=1}^{T_x}a_{t,i}h_i
$$
> 其中，$a_{t,i}$表示第$t$个decoder的隐藏状态对第$i$个encoder的隐藏状态$h_i$的注意力权重。

![](./cs231-img/r.png)

接下来重复这个过程，将下一个时间步的状态$s_1$与Encoder的各个$h_t$输入$f_{att}$,得到$c_2$，以此类推。

因此：

-  输入序列的信息传递不会受单一上下文向量的阻碍

- Decoder的每个时间步都能够“查看”输入序列的不同部分，从而能够生成更高质量的输出序列。

对于计算得到的概率分布$a_{t,i}$进行矩阵可视化，可以看到decoder输出的每个单词关注了输入序列的不同部分：

![](./cs231-img/visu.png)

我们将两种语言的单词进行对应，可以发现attention机制很好地捕捉到了两种语言中同义单词之间的对应关系：

![](./cs231-img/visua1.png)

事实上，attention机制并不关心输入是否是一个顺序序列（ordered sequence ），而是对整个输入序列进行“注意”。

### Image Captioning with RNNs and Attention
---

![](./cs231-img/sps.png)

![](./cs231-img/sssss.png)

![](./cs231-img/s3.png)

![](./cs231-img/seee.png)

### General Attention Layer
---


### The Self-attention Layer
---



### Positional encoding
---


### Masked self-attention layer
---


### Multi-head self-attention layer
---


### Image Captioning using Transformers
---


### The Transformer encoder block
---


### The Transformer decoder block
---


### ViTs– Vision Transformers
---