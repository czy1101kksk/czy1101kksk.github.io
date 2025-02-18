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


```NetworkX```提供了多个类来存储不同类型的图，如有向图和无向图。它还提供了用于创建多重图（有向图和无向图）的类。

```python
import networkx as nx

# Create an undirected graph G
G = nx.Graph()
print(G.is_directed())  # False

# Create a directed graph H
H = nx.DiGraph()
print(H.is_directed())  # True

# Add graph level attribute
G.graph["Name"] = "Bar"
print(G.graph)
```

<B>Node</B>
---

```python
# Add one node with node level attributes
G.add_node(0, feature=5, label=1)

# Get attributes of the node 0
node_0_attr = G.nodes[0] # {'feature': 5, 'label': 1}

G.nodes(data=True) # {0: {'feature': 5, 'label': 1}} 

G.add_nodes_from([
  (1, {"feature": 1, "label": 1}),
  (2, {"feature": 2, "label": 2})
]) # add more nodes through list

# Get number of nodes
num_nodes = G.number_of_nodes()

# Loop through all nodes 
for node in G.nodes(data=True):
    print(node)
```

<B>Edge</B>
---

```python
# Add one edge with edge weight
G.add_edge(0, 1, weight=0.5)

G.add_edges_from([
  (1, 2, {"weight": 0.3}),
  (2, 0, {"weight": 0.1})
])

for edge in G.edges():
  print(edge)

num_edges = G.number_of_edges()

nx.draw(G, with_labels=True)
```

![](./img2/NodeAndEdge.png)

<B>nodes'relation</B>
---

```python
G.degree[1] # node's degree
G.degree() # [(0, 2), (1, 2), (2, 2)]

for neighbor in G.neighbors(1):
    print(neighbor) 
# 0 2
```

<B>PyTorch Geometric (PyG)</B>是关于Pytorch的图深度学习拓展库。

```python
import torch
from torch_geometric.datasets import KarateClub 
# 空手道俱乐部的 34 名成员的社交网络

dataset = KarateClub()
print(len(dataset), dataset.num_features, dataset.num_classes) # 1 34 4
# 数据集包含的图数，节点表示向量的维度，节点类别数

graph = dataset[0]
print(graph)
# Data(x=[34, 34], edge_index=[2, 156], y=[34], train_mask=[34]) 
# train_mask 附加属性，描述了我们已经知道哪些节点的社区分配
print(graph.num_nodes, graph.num_edges) # 34 156


graph.has_isolated_nodes() # False 
graph.has_self_loops() # False
graph.is_undirected() # True 
```


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
\frac{1}{p}, & if \ \  d_{tx}=0 .\\
1, & if \ \ d_{tx}=1  .\\
\frac{1}{q}, & if \ \ d_{tx}=2 .\\
\end{cases}
$$

![](./img/c15.png)
![](./img/c16.png)
![](./img/c17.png)

>Core idea: Embedding nodes so that distances in embedding space reflect node similarities in the original network.

<B>Alias采样</B>

Node2vecWalk中不再是随机抽取邻接点，而是按概率抽取。Alias的核心思想是将一个非均匀分布转化为多个<B>均匀分布的组合</B>，能够加快采样速度，初始化后的采样时间复杂度为$O(1)$，需要存储```accepet```与```alias```两个数组，空间复杂度为$O(2N)$。

给定如下离散概率分布，有$N$个可能发生的事件。每列矩形面积表示该事件发生的概率，柱状图中所有矩形的面积之和为1。

![](./img/alias1.png)

再根据这个矩形，转换成相应的```Accept```表和```Alias```表。

![](./img/alias3.png)

将每个事件的发生的概率乘以$N$，此时会有部分矩形的面积大于1，部分矩形的面积小于1。切割面积大于1的矩形，填补到面积小于1的矩形上，并且每一列至多由两个事件的矩形构成，最终组成一个面积为$1 \times N$的矩形。


![](./img/alias2.png)

首先从$1$~$N$随机生成一个整数i，决定从$1 \times N$矩形中选择第几列，再生成一个均匀随机数$u \in (0,1)$，若若```u < Accept[i]```，则采样```i```对应的事件，否则采样```Alias[i]```。

因为该采样过程不需要根据随机概率在区分度为$N$的线段中寻找，只需要<B>2选1</B>，所以复杂度降低至$O(1)$，这也是其优于传统采样的原因。

![](./img/node2vec.png)

![](./img/c18.png)

### Matrix Factorization

![](./img/matrix.png)
![](./img/c19.png)
![](./img/c20.png)


在Deepwalk和Node2Vec中，我们通过随机游走得到节点序列，使得节点相似度（node similarity）的定义更加复杂



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