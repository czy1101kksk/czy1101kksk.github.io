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


### Knowledge Graphs

![](./img2/13.png)

![](./img2/14.png)

![](./img2/15.png)

![](./img2/16.png)

![](./img2/19.png)

![](./img2/17.png)

![](./img2/18.png)


###

![](./img2/20.png)

![](./img2/21.png)

![](./img2/22.png)

![](./img2/23.png)

![](./img2/24.png)

![](./img2/25.png)

![](./img2/26.png)

![](./img2/27.png)

![](./img2/28.png)

![](./img2/29.png)

![](./img2/30.png)

### TransE

For a triple $(h,r,t)$,let $h,r,t \in \mathbb{R}^d$ be embedding vectors.

<B>TransE</B>: $h + r ≈ t$ if the given link exists else $h + r ≠ t$.

Entity scoring func:

$$
f_r(h,t) =  -|| h + r - t ||
$$

![](./img2/22.png)

![](./img2/23.png)

> 对比损失(Contrastive loss)：对有效的三元组支持较低的距离（或较高的分数），对损坏的三元组则支持较高的距离（或者较低的分数）

### Connectivity Patterns in KG

- Symmetry:  If the edge $(h,"Roommate",t)$ exists in KG, then the edge $(t,"Roommate",h)$ should also exist.

- Inverse relation :  If the edge $(h,"Advisor",t)$ exists in KG, then the edge $(t, "Advisee",h)$ should also exist.

> Are TransE expressive enough to capture these patterns?

![](./img2/24.png)
![](./img2/25%20(1).png)
![](./img2/25%20(2).png)
![](./img2/25%20(3).png)
![](./img2/25%20(4).png)
![](./img2/26.png)

### TransR

> TransE models translation of any relation in the same embedding space.

<B>TransR</B>: model entities as vectors in the entity space $\mathbb{R}^d$ and model each relation as vector in relation space $\mathbf{r} \in \mathbb{R}^{k}$ with $\mathbf{M}_r \in \mathbb{R}^{k \times d}$ as the projection matrix.

$h_{k} = M_r h, t_k=M_r t$

scoring func:
$$ 
f_r(h,t) = -|| h_k + r - t_k ||
$$

![](./img2/27.png)
![](./img2/27%20(1).png)
![](./img2/27%20(2).png)
![](./img2/27%20(3).png)
![](./img2/27%20(4).png)

- DistMult

Entities and relations are vectros in $\mathbb{R}^k$

Score func:

$$
f_r(h,t)= \sum_i \mathbf{h}_i \cdot \mathbf{r}_i \cdot \mathbf{t}_i
$$

![](./img2/28.png)

> Intuition of the score function: Can be viewed as a cosine similarity between $\mathbf{h} \cdot \mathbf{r}$ and $\mathbf{t}$

![](./img2/29%20(1).png)

![](./img2/29%20(2).png)

![](./img2/30.png)

![](./img2/31.png)

![](./img2/32.png)

- ComplEx

model entities and relations as complex vectors in $\mathbb{C}^k$

![](./img2/33.png)

Score func:

$$
f_r(h,t) = \text{Re}(\sum_i \mathbf{h}_i \cdot \mathbf{r}_i \cdot \bar{\mathbf{t}_i} )
$$

![](./img2/34%20(1).png)

![](./img2/34%20(2).png)

![](./img2/34%20(3).png)

![](./img2/34%20(4).png)

![](./img2/35.png)