# 🔗<B>Chapter 3：Recurrent neural network</B>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>

## 自回归模型 autoregressive models
---

自回归模型是统计上一种处理时间序列的方法，用同一变数例如$x$的之前各期，亦即$x_1$至$x_{t-1}$来预测本期$x_t$的表现，并假设它们为线性关系。因为这是从回归分析中的线性回归发展而来，只是不用$x$预测$y$，而是用$x$预测$x$（自己）；所以叫做自回归。

\[
    X_t = \sum_{i=1}^p \phi_i X_{t-i} + c + \epsilon_t        
\]

第二种如下图，是保留一些对过去观测的总结$h_t$, 并且同时更新预测$\hat{x}_t$和总结$h_t$。即为基于\hat{x}_t = P(x_t \mid h_{t})$来估计$x_t$，以及通过$h_t = g(h_{t-1}, x_{t-1})$更新总结$h_t$的模型，称为隐变量自回归模型（latent autoregressive models）。

![](./d2l-img/ar1.png)

## 词元化
---



## 循环神经网络
---

![](./d2l-img/recu.png)

假设在时间步$t$有小批量输入$\mathbf{X}_t \in \mathbb{R}^{n \times d}$，用$\mathbf{H}_t \in \mathbb{R}^{n \times h}$表示时间步$t$的隐藏变量。对于RNN网络，我们保存上一步的隐藏变量$\mathbf{H}_{t-1}$，引入一个新的权重$\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$，则有：

\[
    \mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}  + \mathbf{b}_h)    
\]

!!! advice "```nn.RNN()```"

    <font size=3>
    ```nn.RNN(input_size=, hidden_size=, num_layers=, )``` 

    - ```input_size (int)```：输入数据的特征大小(特征维度)。即每个时间步的输入向量$x_t$的维度。

    - ```hidden_size (int)```：隐藏层的特征大小，即每个时间步的隐藏状态向量ht的维度。它决定了模型的表示能力和记忆能力。较大的`hidden_size`通常允许模型学习更复杂的模式，但也需要更多的计算资源。

    - ```num_layers (int)```:RNN的层数，用于堆叠多个RNN层，默认值为1。当层数大于1时，RNN会变为多层RNN。多层RNN可以捕捉更复杂的时间依赖关系，但也会增加模型的复杂性。

    - ```nonlinearity (str)```:指定激活函数，默认值为'tanh'。可选值有'tanh'和'relu'。
    
    - ```bias (bool)```:如果设置为True，则在RNN中添加偏置项。默认值为True。偏差项通常有助于模型更好地拟合数据。

    - ```dropout (float)```:如果非零，则在除最后一层之外的每个RNN层之间添加dropout层，其丢弃概率为dropout。默认值为0。这有助于防止过拟合。
    
    - ```bidirectional (bool)```:一个布尔值，确定是否使用双向RNN。如果设置为True，RNN将同时在时间步的正向和反向方向上运行（则使用双向RNN），以捕捉前后的上下文信息。默认值为False。


    </font>