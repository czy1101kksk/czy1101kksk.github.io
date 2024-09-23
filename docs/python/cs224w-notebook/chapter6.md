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

###  Heterogeneous graphs(异构图)

> 异质图(heterogeneous graph，也称异构图)，又被称为异质信息网络(heterogeneous information network，也称异构信息网络)。区别于同质图(homogeneous graph)，它是一种具有多种节点类型或多种边类型的图数据结构( graphs with multiple nodes or edge types)，用于刻画复杂异质对象及其交互，具有丰富的语义信息，为图数据挖掘提供了一种有效的建模工具和分析方法。同时，异质图数据也是一种广泛存在的数据类型，例如知识图谱，社交网络数据。

- Relational GCNs

- Heterogeneous Graph Transformer

- Design space for heterogeneous GNNs

![](./img/tu1.png)

![](./img/tu2.png)

A heterogeneous graph can be defined as :

$$
\mathbf{G} = \{\mathbf{V}, \mathbf{E}, \tau,\phi \}
$$

- Node with node types: $v \in \mathbf{V}$

- Node types for node: $v:\tau (v)$

- Edge with edge types: $(u,v) \in \mathbf{E}$

- Edge types: $\(u,v): \phi (u,v)$

heterogeneous graph's relation type for edge $e$ is a tuple:$r(u,v) = (\tau(u,v), \phi(u,v), \tau(v))$

![](./img/tu3.png)

### Relation GCN(RGCN)

> How do we extend GCNmodel to handle heterogeneous graphs?

![](./img/tu4.png)

$$
h^{(l+1)}_v = \sigma( \sum_{r \in R} \sum_{u \in N^r_v} \frac{1}{c^{r}_{v}} W_r^{(l)} h^{(l)}_u + W_0^{(l)} h^{(l)}_v )
$$

- $c^{r}_{v}$ is the number of relations of $v$, for normalization.

- Each neighbor of a given relation:

$$
m_{u,r}^{(l)} = \frac{1}{c_{v,r}} W_r^{(l)} h^{(l)}_u
$$

- Self-loop"

$$
m_{v}^{(l)} = W_0^{(l)} h^{(l)}_v
$$

- Aggregation:

$$
h^{(l+1)}_v = \sigma(Sum(\{{m_{u,r}^{(l)} ， u \in N^r_v}\}) ∪ \{m_{v}^{(l)}\})
$$

![](./img/tu5.png)

### Block Diagonal Matrix

![](./img/tu6.png)

![](./img/tu7.png)

