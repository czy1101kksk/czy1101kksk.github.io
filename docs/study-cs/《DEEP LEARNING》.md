# 🛠 《DEEP LEARNING》阅读笔记

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>

!!! info "相关信息"
    <font size =4>
    正好ZJUAI协会有组队学AI的活动,面向我这种理论基础比较薄弱的同学,而且我从大一下开始学DL一直处于一个人闭门造车的阶段,找不到水平相近的人交流,所以毫不犹豫参加了组队学习的活动🤗
    
    - 书籍:《DEEP LEARNING》
    
    - 作者：[Ian Goodfellow](https://scholar.google.com/citations?user=iYN86KEAAAAJ&hl=zh-CN&oi=ao), [Yoshua Bengio](https://scholar.google.com/citations?hl=zh-CN&user=kukA0LcAAAAJ), [Aaron Courville](https://scholar.google.com/citations?hl=zh-CN&user=km6CP8cAAAAJ)
    - 电子书链接：https://www.deeplearningbook.org/
    </font>


## Part I: Applied Math and Machine Learning Basics

### Lecture 1: Linear Algebra
---

- 张量(tensor): 当一个数组的元素分布在若干维度坐标的规则网络中,称之为张量,可记作$\mathit{A_{i,j,k}}$

<B>转置</B>(transpose):以对角线为轴对矩阵进行镜像: $\mathit{(A^{\mathsf{T}})_{i,j}} = \mathit{A_{j,i}}$

```python
import torch

ExmapleTensor = torch.randn(2,3)
TransposedTensor = torch.transpose(ExmapleTensor) 
# torch.transpose(ExmapleTensor, 0, 1) 将第0维和第1维进行交换
```

!!! note "<B>广播机制</B>"
    <font size = 4>
    在深度学习中,当矩阵运算形状不匹配时,允许在执行元素级操作时自动扩展数组的维度(<B>一般用于矩阵与单个向量之间的运算</B>)Broadcasting机制使用的前提是两个参与运算的张量是可broadcastable的

    <B>Broadcastable</B>
    
    - 每个tensor至少是一维的

    - 两个tensor的维数从后往前对应，对应位置要么是相等的，要么其中一个是1，或者不存在
    
    常用示例:
    ```python
    import torch

    BroadcastingTensor_1 = torch.tensor([1, 2, 3]) + torch.tensor([2])  
    #: tensor([3, 4, 5])
    
    BroadcastingTensor_2 = torch.tensor([[1, 2], [3, 4]]) + torch.tensor([1, 2]) 
    #: tensor([[2, 4],
    #          [4, 6]])
    
    BroadcastingTensor_3 = torch.tensor([[1, 2], [3, 4]]) + torch.tensor([[1], [2]]) 
    #: tensor([[2, 3],
    #          [5, 6]])

    BroadcastingTensor_multiply = torch.tensor([[1, 2], [3, 4]]) * torch.tensor([1, 2])
    #: tensor([[1, 4],
    #          [3, 8]])
    ```
    </font>

<B>矩阵乘法</B>:
---

在进行矩阵乘法(matrix multiplication)时,矩阵$\mathit{A}$的行数与矩阵$\mathit{B}$的行数相等:

\[
    \mathit{C_{m \times p}} = \mathit{A_{m \times n}} \mathit{B_{n \times p}}
\]

!!! note "Hadamard乘积"
    <font size = 3.5>
    Hadamard乘积（矩阵逐元素乘积）是指两个形状相同的张量之间的元素对应相乘的操作。这种操作在深度学习中经常用于实现多种功能(特征融合),记为:
    
    \[
        \mathit{C} = \mathit{A} \odot \mathit{B}    
    \]
    
    
    ```python
        Tensor1 = torch.tensor([[1, 2],[3, 4]])
        Tensor2 = torch.tensor([[3, 4],[6, 2]])
        result = torch.mul(Tensor1, Tensor2)
        #:tensor([[ 3,  8],
        #         [18,  8]])
    ```
    </font>

<B>单位矩阵$I_n$</B>:
---

单位矩阵(identity matrix)所有沿着主对角线的元素都是1,而其他元素均为0

\[
    I_n=
    \begin{eqnarray}
    \begin{pmatrix}
        1 & \cdots & 0 \\
        \vdots & \ddots & \vdots
        \\0 & \cdots & 1
    \end{pmatrix}
    \end{eqnarray}
\]

对于单位矩阵$I_n$,任意向量与$I_n$相乘不会变化, 形式上: $I_n \in \mathbb{R}^{n \times n} , \forall x \in \mathbb{R}^{n}, I_n x = x$  

```python
IdentityTensor = torch.eye(3)
#: tensor([[1., 0., 0.],
#          [0., 1., 0.],
#          [0., 0., 1.]])
```

<B>矩阵逆(matrix inversion)</B>:矩阵$\mathit{A}$的矩阵逆记作$\mathit{A^{-1}}$,有:

\[
    \mathit{A} \mathit{A^{-1}} = I_n
\]

```python
A = torch.tensor([[1., 2., 3.], [1.5, 2., 2.3], [.1, .2, .5]])
A_inv = torch.linalg.inv(A) 
# torch.linalg.inv()有当张量是方阵且有非零行列式时，才能计算逆
print("逆矩阵:", A_inv)
#: 逆矩阵: tensor([[ -2.7000,   2.0000,   7.0000],
#                 [  2.6000,  -1.0000, -11.0000],
#                 [ -0.5000,   0.0000,   5.0000]])

>>> B = torch.tensor([[1., 2.], [3., 4.]])      # 输入一个不可逆矩阵,会报错
>>> torch.linalg.inv(B)
_LinAlgError: linalg.inv: The diagonal element 2 is zero, the inversion could not be completed because the input matrix is singular.
>>> B_int = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64) # 输入矩阵的元素类型不支持
>>> torch.linalg.inv(a)
RuntimeError: Input tensor must be a floating point or complex tensor.
```


<B>范数</B>:
---
在深度学习中我们经常用$L^p$范数来衡量一个向量的大小,记作:

\[
    \lVert x \Vert_p =  ( \sum_{i=1}^n |x_i|^p )^{\frac{1}{p}}    
\]

- <B>$L^1$范数</B>(曼哈顿距离): 向量元素绝对值的总和,在零与非零元素之间的差异非常重要时使用$L^1$范数

\[
    \lVert x \Vert_1 =   \sum_{i} |x_i|     
\]

- <B>$L^2$范数</B>(欧几里得范数): 表示从远点出发到向量$x$确定的点的欧几里得距离,是最常用的范数类型

\[
    \lVert x \Vert_2 =  \sqrt{ \sum_{i} |x_i|^2 }     
\]

- <B>$L^∞$无穷范数</B>:无穷范数是向量中所有元素绝对值的最大值,度量了向量中的最大元素，对异常值非常敏感.

\[
    \lVert x \Vert_∞ = \mathop{max}\limits_{i} (|x_i|)       
\]

- <B>$Frobenius$范数</B>:矩阵元素平方和的平方根,度量了矩阵的整体“能量”，适用于矩阵而非向量。(类似用于矩阵的$L^2$范数)

\[
    \lVert A \Vert_F =  \sqrt{ \sum_{i,j} A_{i,j}^2 }     
\]

<B>正交矩阵</B>
---
若存在向量$x$与$y$,满足$x^T y = 0$,则向量$x$与$y$相互正交(orthogonal).正交矩阵是一个方阵，它的行向量和列向量都是<B>标准正交的</B>，即它们的内积为零，而每个向量的范数为一($\lVert x \Vert_2 = 1$)。具体来说，如果矩阵 \( Q \) 是一个 \( n \times n \) 的正交矩阵，那么它满足以下条件：

\[ Q^TQ = QQ^T = I_n \]

由此可得:

- **逆矩阵**：正交矩阵的逆矩阵就是它的转置，即\( Q^{-1} = Q^T \)。

- **行列式**：正交矩阵的行列式的绝对值为 1，即 \( |\det(Q)| = 1 \)。


<B>特征分解</B>
---

对于一个给定的方阵 \( A \)，如果存在一个非零向量 \( v \) 和一个标量 \( \lambda \)，使得下面的等式成立：

\[ A \cdot v = \lambda \cdot v \]

那么 \( \lambda \) 被称为矩阵 \( A \) 的一个特征值（Eigenvalue），而 \( v \) 被称为对应的特征向量（Eigenvector）。

特征分解的目的是将矩阵 \( A \) 表示为一组<B>特征值和特征向量的乘积形式</B>。对于非奇异矩阵，特征分解可以写成：

\[ A = V \cdot diag(\lambda) \cdot V^{-1} \]

其中，\( V \) 是一个包含 \( A \) 所有特征向量的矩阵，\( diag(\lambda) \) 是一个对角矩阵，其对角线上的元素是对应的特征值。注意，不是所有矩阵都有特征分解，<B>只有方阵才可能有特征分解</B>。

```python
# 创建一个 3x3 矩阵
A = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]], dtype=torch.float64)

# 计算特征值和特征向量
eigenvalues, eigenvectors = torch.linalg.eig(A)

# 输出特征值和特征向量
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# 验证特征分解
reconstructed_A = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
print("Reconstructed A:\n", reconstructed_A)

#: Eigenvalues: tensor([ 1.6117e+01+0.j, -1.1168e+00+0.j, -1.3037e-15+0.j],dtype=torch.complex128)
#: Eigenvectors:
#: tensor([[-0.2320+0.j, -0.7858+0.j,  0.4082+0.j],
#          [-0.5253+0.j, -0.0868+0.j, -0.8165+0.j],
#          [-0.8187+0.j,  0.6123+0.j,  0.4082+0.j]], dtype=torch.complex128)
#: Reconstructed A:
#: tensor([[1.0000+0.j, 2.0000+0.j, 3.0000+0.j],
#          [4.0000+0.j, 5.0000+0.j, 6.0000+0.j],
#          [7.0000+0.j, 8.0000+0.j, 9.0000+0.j]], dtype=torch.complex128)
```

不是每一个矩阵都能分解为特征值与特征向量,并且可能会复数.对于实对称矩阵$A$,每个$A$都能分解为实特征向量与实特征值:

\[ A = Q \Lambda Q^T \]

特征值$\Lambda_{i,i}$对应的特征向量是矩阵$Q$的第i列,记作$Q_{:,i}$.





























!!! note "torch.linalg模块"