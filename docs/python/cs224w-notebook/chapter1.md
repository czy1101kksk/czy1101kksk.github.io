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

    Optional Readings：
    
    [DeepWalk: Online Learning of Social Representations](https://arxiv.org/pdf/1403.6652)

    [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/pdf/1607.00653)

    [Network Embedding as Matrix Factorization](https://arxiv.org/pdf/1710.02971.pdf)
    
    </font>


## Node Embeddings
---

![](./img/c1.png)

在传统机器学习流程中，我们需要对原始数据进行特征工程```feature engineering```（比如提取特征等），但是现在我们使用表示学习```representation learning```的方式来自动学习到数据的特征，直接应用于下游预测任务。

图的表示学习：Map nodes into an embedding space, similarity of embeddings between nodes indicates their similarity in the network.For exmaple : Both nodes are close to each other(connected by an edge).

![](./img/c2.png)

![](./img/c6.png)

we assume an graph $G$: $V$ is the vertex set and $A$ is the adjacency matrix (assume binary).

> the adjacency matrix(邻接矩阵): 表示顶点之间相邻关系的矩阵，若两个顶点相邻，则对应位置的值为1，否则为0。

![](./img/c3.png)

Our goal is to encode nodes making the similarity in the embedding space reflect the similarity in the graph.

![](./img/c4.png)

So <B>the definition of similarity function</B> is the key(a 
measure of similarity in the original network).

$$
similarity(u,v)  \approx  \mathbf{z}_v^T \mathbf{z}_u
$$

<B>Encoder</B>: maps each node to a low-dimensional vector $\mathbf{z}_v$.we assume $Z \in \mathbb{R}^{d \times |V|}$ as the matrix of embeddings and $v \in \mathbb{I}^{|V|}$ as indicator vector.

$$
ENC(v) = \mathbf{z}_v \stackrel{simplest}{=} \mathbf{Z} \cdot v
$$

![](./img/c5.png)

### Random Walks

![](./img/c7.png)

> $P(v|\mathbf{z}_u)$是从$u$开始随机游走能到$v$的概率，衡量$u$和$v$的相似度，用节点embedding向量相似性算概率。 

![](./img/c8.png)

$$
\mathbf{z}^T_u \mathbf{z}_v = \text{the probability that u and v co-occur on a random walk over the graph}
$$

![](./img/c9.png)

> Why random walks?
    
    - Expressivity: Flexible stochastic definition of node similarity that incorporates both local and higher-order neighborhood information
    
    - Idea: if random walk starting from node 𝒖 visits 𝒗 with high probability, 𝒖 and 𝒗 are similar (high-order multi-hop information)
    
    - Efficiency: Do not need to consider all node pairs when training; only need to consider pairs that co-occur on random walks

The definition of nearby nodes and our goal to learn a mapping:

- $N_R(u)$: neighbourhood of $u$ which can be obtained by random walk

- $f:u → \mathbb{R}^d:f(u)=\mathbf{z}_u$ 

-  Log-likelihood objective:

    $\mathop{\arg\max}\limits_{z} \mathop{\sum}\limits_{u \in V}  \log P(N_R(u) | \mathbf{z}_u) $

    Equivalently,

    $\mathop{\arg\min}\limits_{z} ℒ = \mathop{\sum}\limits_{u \in V} \mathop{\sum}\limits_{v \in N_R(u)} - \log P(v | \mathbf{z}_u)$

    > 使相邻的nodes之间的相似度最大化

![](./img/c10.png)

- Parameterize $P(v|\mathbf{z}_u)$ as a softmax function:

    $$
    P(v|\mathbf{z}_u) = \frac{\exp(\mathbf{z}_u^T \mathbf{z}_v)}{\sum_{n \in V} \exp(\mathbf{z}_{u}^T \mathbf{z}_n)}
    $$

so,the log-likelihood objective can transfer to:

$$
ℒ = \mathop{\sum}\limits_{u \in V} \mathop{\sum}\limits_{v \in N_R(u)} - \log \frac{\exp(\mathbf{z}_u^T \mathbf{z}_v)}{\sum_{n \in V} \exp(\mathbf{z}_{u}^T \mathbf{z}_n)}
$$

![](./img/c11.png)

> Obviously $O(|V|^2)$ complexity, doing this naively is too expensive.

![](./img/c12.png)

![](./img/c13.png)

> 层次Softmax（Hierarchical Softmax）优化算法，避免计算所有词的softmax

> 上述的随机游走策略是完全随机的，固定长度的游走，是否需要改进？

### DeepWalk：RandWalk + Skip-Gram
---

Code：[https://github.com/phanein/deepwalk](https://github.com/phanein/deepwalk)

DeepWalk将图数据与自然语言处理技术（Word2Vec）相结合，通过随机游走将图结构转化为节点序列，然后使用Skip-Gram模型训练词嵌入，用于学习网络中顶点的潜在表示。

![](./img/deepwalk.png)

①从网络中的每个节点开始分别进行RandomWalk采样，得到局部相关联的训练数据；

②对采样数据进行SkipGram训练，将离散的网络节点表示成向量化，最大化节点共现，使用Hierarchical Softmax来做超大规模分类的分类器



### Node2Vec：Biased Walks
---

Code：https://github.com/aditya-grover/node2vec

> Node2Vec和随机游走的区别是如何定义相邻节点集——以及如何定义随机游走的策略(偏随机游走)

![](./img/c14.png)

- BFS：节点功能角色structural equivalence

- DFS：同质社群homophily


Node2Vec gives two parameters to control the random walk:

- Return parameter $p$: probability of returning to the previous node

- In-out parameter $q$: the “ratio” of BFS vs. DFS

引入这两个超参数$p，q$，来控制随机游走的策略。假设当前随机游走经过边$(t,v)$到达节点$v$。则转移策略遵循以下公式：$\pi_{vx}=\alpha_{pq}(t,x) \cdot w_{vx}$，转移策略为$\alpha_{pq}(t,x)$，$w_vx$是节点$v$与$x$之间的边权。$d_{tx}$为节点$t$和$x$之间的最短路径距离：

$$
\alpha_{pq}(t,x) = 
\begin{cases}
\frac{1}{p}, & if \quad d_{tx}=0 \\
1, & if \quad d_{tx}=1  \\
\frac{1}{q}, & if \quad d_{tx}=2 \\
\end{cases}
$$

![](./img/c15.png)
![](./img/c16.png)
![](./img/c17.png)

>Core idea: Embedding nodes so that distances in embedding space reflect node similarities in the original network.

![](./img/c18.png)

### Matrix Factorization

![](./img/c19.png)
![](./img/c20.png)

### Limitations

- 无法立刻泛化到新加入的节点，无法处理动态网络（Cannot obtain embeddings for nodes not in the training set. Cannot apply to new graphs, evolving graphs）

- Cannot capture structural similarity

![](./img/c21.png)

- 仅仅使用了节点之间的连接信息(Cannot utilize node, edge and graph features)

### Embedding Entire Graphs

The Goal: Embed a subgraph(子图) $G$ into a low-dimensional space $\mathbb{R}^d$

- Approach 1: 直接对所有节点键入求和/平均

$$
z_G = \sum_{v \in G} z_v
$$

- Approach 2: 引入一个虚拟节点（virtual node），求出虚拟节点的嵌入来代替子图的嵌入

![](./img/a1.png)

- Approach 3: Anonymous Walks(匿名随机游走)


![](./img/c22%20(1).png)

![](./img/c22%20(2).png)