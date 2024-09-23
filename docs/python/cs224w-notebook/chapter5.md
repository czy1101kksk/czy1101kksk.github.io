# 🛣[Deep Learning]Stanford CS224w:Machine Learning with Graphs
---
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>

!!! info "想说的话🎇"
    <font size = 3.5>
    
    🔝课程网站：http://web.stanford.edu/class/cs224w/
    
    👀一些资源: 
    B站精讲：https://www.bilibili.com/video/BV1pR4y1S7GA/?spm_id_from=333.337.search-card.all.click&vd_source=280e4970f2995a05fdeab972a42bfdd0
    
    https://github.com/TommyZihao/zihao_course/tree/main/CS224W
    
    Slides: http://web.stanford.edu/class/cs224w/slides
    
    </font>

![](./img/u1.png)

### Neighborhood Aggregation

- Observation: Neighbor aggregation can be abstracted as a function over a multi-set (a set with repeating elements).

![](./img/u2.png)

![](./img/u4.png)

- GCN

![](./img/u5.png)

- GraphSAGE

![](./img/u6.png)

![](./img/u7.png)

![](./img/mm.png)
![](./img/666%20(4).png)
![](./img/666%20(5).png)

![](./img/u8.png)

### Designing Most Expressive GNNs

![](./img/u9.png)

![](./img/u7%20(1).png)

![](./img/u7%20(2).png)

### Graph Isomorphism Network(GIN)

GIN‘s neighbor aggregation function is <B>injective</B>, so GIN is the most expressive GNN

![](./img/u10.png)

- 1-Weisfeiler-Lehman（Color refinement algorithm）算法

![](./img/uu.png)

![](./img/666%20(1).png)
![](./img/666%20(2).png)
![](./img/666%20(3).png)

GIN uses a NN to model the injective HASH function

$$
\begin{aligned}
&GINconv(c^{(k)}(v),\{ c^{(k)}(u)_{u \in N(v)} \}) \\
=& MLP_{\phi} \Big( (1+\epsilon)MLP_f (c^{(k)}(v))+ \sum_{u\in N(v)} MLP_f (c^{(k)}(u)) \Big)\\
\end{aligned}
$$

> where $\epsilon$ is a learnable parameter.

![](./img/aaa.png)

![](./img/45.png)

![](./img/22.png)

### General tips

![](./img/33.png)

### Understand GIN 

论文地址：[How Powerful are Graph Neural Networks](https://arxiv.org/abs/1810.00826)

