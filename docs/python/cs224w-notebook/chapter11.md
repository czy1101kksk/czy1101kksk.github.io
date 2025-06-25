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

### Geometric Graphs
---

A geometric graph $G=(A,S,X)$ is a graph where each node is embeddedd in $d$-dimensional Euclidean space:

![](./img2/geo.png)

- $A$: an $n \times n$ adjacency matrix

- $S \in \mathbb{R}^{n \times f}$: scalar features

- $X \in \mathbb{R}^{n \times d}$: tensor features(e.g.,coordinates)

![](./img2/geo1.png)

![](./img2/geo3.png)

![](./img2/geo2.png)

![](./img2/geo4.png)

![](./img2/geo5.png)

![](./img2/geo6.png)

### Geometric GNNs

- <B>Invariant GNNs</B> for learning invariant scalar features

- <B>Equivariant GNNs</B> for learning equivariant tensor features

Invariant GNNs: <B>SchNet</B>
---

[]()