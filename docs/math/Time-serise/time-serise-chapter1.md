#  Time Serise learning
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
!!! info "Override🎇"
    
    <font size = 3.5>
    
    课本：《时间序列与机器学习》（张戎、罗齐）、《应用时间序列》（何书元）
    
    </font>

!!! note "时间序列"
    
    <font size = 3.5>

    按照<B>时间次序</B>排列的随机变量序列称为<B>时间序列</B>：

    \[
        X_1, X_2,......
    \]

    若$x_1,x_2,...x_N$表示随机变量的观测值，则称之为时间序列(1, 1)的N个观测样本。

    若存在两个时间序列${X_t}$与${Y_t}$之间存在相关关系，则可表示为：
    
    \[
        \xi_t = (X_t,Y_t)^T, t=1,2,...
    \]

    这个向量值的时间序列称为<B>多维时间序列</B>
    </font>

### 时间序列的分解
---

大量的时间序列的观测样本体现出:

- <B>趋势性${T_t}$</B>

- <B>季节性${S_t}$</B>,要求$\sum_{j=1}^s S(t+j)=0, t=1,2,...$

- <B>随机性${R_t}$</B>，要求$E(R_t)=0, t=1,2,...$

因此，时间序列的分解可以表示为：

\[
    X_t = T_t + S_t + R_t, t=1,2,...
\]

将时间序列的趋势项、季节项和随机项分解出来的工作称之为<B>时间序列分解</B>











