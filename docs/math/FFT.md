# 快速傅里叶变换（FFT）

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>

### 傅里叶级数

对于周期为$T$的函数$f(t)$，傅里叶级数的目标是将它表示为一系列复指数函数的线性组合：

$$
f(t) = \sum_{n=-\infty}^{\infty} c_n e^{j n \omega_0 t}，w_0 = \frac{2\pi}{T}
$$

其中，$e^{j n \omega_0 t}= \cos{(n \omega_0 t)} + j \sin{(n \omega_0 t)}$是基函数，满足正交性（不同频率的复指数函数在特定区间上的积分结果为零，而同一频率的积分结果为非零值（通常为周期长度）：

$$
\int_{-T/2}^{T/2} e^{j n \omega_0 t} \cdot e^{-j m \omega_0 t} dt = \delta(n-m) \cdot T = 
\begin{cases}
T, &  n = m \\
0, &  n \neq m
\end{cases}
$$

对$f(t) = \sum_{n=-\infty}^{\infty} c_n e^{j n \omega_0 t}，w_0 = \frac{2\pi}{T}$，在等式两边乘以$e^{j -m \omega_0 t}$并且积分：

$$
\int_{-T/2}^{T/2} f(t) e^{-j m \omega_0 t} dt = \sum_{n=-\infty}^{\infty} c_n \int_{-T/2}^{T/2} e^{j n \omega_0 t} \cdot e^{-j m \omega_0 t} dt 
$$

根据正交性，右边仅当$n=m$时积分值为$T$，其余为0：

$$
\int_{-T/2}^{T/2} f(t) e^{-j m \omega_0 t} dt = c_m \cdot T
$$

$$
c_m = \frac{1}{T} \int_{-T/2}^{T/2} f(t) e^{-j m \omega_0 t} dt
$$

系数$c_n$表示信号$f(t)$中频率为$n \omega_0$的分量的强度，因此傅里叶级数是将将信号分解为基频$\omega_0$以及谐波（$2\omega_0,3\omega,...$）的线性组合。

而对于非周期函数，我们把它的周期看作无穷大$T \rightarrow \infty$，则基频$\omega_0=2\pi/T \rightarrow 0$，离散频率$nω_0$变为连续变量$ω$。

$$
c_n = \frac{\omega_0}{2\pi} \int_{-T/2}^{T/2} f(t) e^{-j n \omega_0 t} dt
$$

当$T \rightarrow \infty$，令$\omega=n\omega_0，\omega_0 = d\omega$:

$$
\begin{aligned}
f(t) =& \lim_{T \rightarrow \infty} \sum_{n=-\infty}^{\infty} [\frac{\omega_0}{2\pi} \int_{-T/2}^{T/2} f(\tau) e^{-j n \omega_0 \tau} d\tau] e^{j n \omega_0 t} \\
=& \frac{1}{2\pi} \int_{-\infty}^{\infty} [\int_{-\infty}^{\infty} f(\tau) e^{-j \omega \tau} d\tau]e^{j \omega t} d\omega
\end{aligned}
$$

因此我们得到非周期函数的傅里叶变换及其逆变换：

$$
\begin{aligned}
F(\omega) =& \int_{-\infty}^{\infty} f(t) e^{-j \omega t} dt \\
f(t) =& \frac{1}{2\pi} \int_{-\infty}^{\infty} F(\omega) e^{j \omega t} d\omega
\end{aligned}
$$

### 傅里叶变换

傅里叶变换的公式为：

$$
F(\omega) = \int_{-\infty}^{\infty} f(t) e^{-j\omega t} dt
$$

- 输入：时域信号$f(t)$（例如随时间变化的电压、声波）

- 输出：频谱密度函数，该函数的自变量是$\omega=\frac{2\pi}{T}$，因变量是信号幅值在频域中的分布密度，即单位频率信号的强度。

### DFT

离散傅里叶变换（Discrete Fourier Transform, DFT）是傅里叶变换的离散化版本，专门用于处理数字信号（如计算机采集的时域数据）。它将有限长度的离散时域信号转换为离散频域表示，揭示信号中隐含的频率成分。

引入梳状函数，其中$T_s$为采样周期：

$$
\delta_s (t) = \sum_{n=-\infty}^{\infty} \delta(t-nT_s)
$$

将时域上的连续信号与它相乘，即可得到：

$$
x_s(t) = x(t) \cdot \delta_s(t) = \sum_{n=-\infty}^{\infty} x(t) \delta(t-nT_s)
$$

根据傅里叶变换，完成时域离散化：

$$
X(\omega) = \int_{-\infty}^{\infty} x(t) e^{-j\omega t} dt
$$

$$
\begin{aligned}
X_s(\omega) &= \int_{-\infty}^{\infty} [\sum_{n=-\infty}^{\infty} x(t) \delta(t-nT_s) e^{-j\omega t}] dt \\
&= \sum_{n=-\infty}^{\infty} \int_{-\infty}^{\infty} x(t) \delta(t-nT_s) e^{-j\omega t} dt\\
&= \sum_{n=-\infty}^{\infty} x(nT_s) e^{-j\omega nT_s}\\
\end{aligned}
$$

显然计算机只能对于连续信号$x(t)$进行有限的N次的采样，采样周期为$T_s$，我们对采样得到的信号进行时域上的周期延拓，这样我们就得到了一个周期为$T_0=NT_s$
 的函数。对于周期函数而言，其频谱密度函数是离散化的，这样就把频域也进行了离散化。

 在一个周期$T_0=NT_s$内：

$$
x_s (t) = \sum_{n=0}^{N-1} x(t) \delta(t-nT_s)
$$

因此离散信号的傅里叶级数($\omega_0=2\pi/T_0$):

$$
\begin{aligned}
X(k\omega_0) &= \frac{1}{T_0} \int_{0}^{T} (\sum_{n=0}^{N-1} x(t) \delta(t-nT_s)) e^{-jkw_0t} dt \\
&= \frac{1}{T_0} \sum_{n=0}^{N-1} \int_{0}^{T} x(t) \delta(t-nT_s) e^{-jkw_0t} dt\\
&= \frac{1}{T_0} \sum_{n=0}^{N-1} x(nT_s) e^{-jkw_0nT_s}\\
&= \frac{1}{NT_s} \sum_{n=0}^{N-1} x(nT_s) e^{-j \frac{2\pi}{NT_s} knT_s }\\
&= \frac{1}{NT_s} \sum_{n=0}^{N-1} x(nT_s) e^{-j \frac{2\pi}{N} kn }
\end{aligned} 
$$

令$X[k] = X(k\omega_0) T_0，x[n] = x(nT_s)$:

$$
X[k] = \sum_{n=0}^{N-1} x[n] e^{-j \frac{2\pi}{N} kn }，(k=0,1,...,N-1)
$$

要计算特定的k值的$X[k]$，要进行N次$x[n]$与$e^{-j \frac{2\pi}{N} kn }$的乘法运算，再进行N-1次加法运算。因此，DFT的计算复杂度为$O(N^2)$。

### FFT

利用复数单位根$e^{-j \frac{2\pi}{N} kn }$的周期性质，FFT可以将DFT的计算复杂度降低到$O(N \log N)$。

