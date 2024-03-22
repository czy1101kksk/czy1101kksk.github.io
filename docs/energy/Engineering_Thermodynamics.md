# 🍥工程热力学(甲)
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
!!! info "🌇Information"
    <font size =4>
    
    - 课程:工程热力学(甲) Engineering Thermodynamics(A)
    
    - 学分:4.0             课程代码: 59120030
    
    - 教师:俞自涛老师       教材:<工程热力学(第三版)> 
    
</font>
<font size = 3.5>

>虽然不是慧能班的,但在无意中选到了慧能班的课,自己和俞老师还算比较熟悉,班上一共只有20多人所以听课体验还不错(某些百人的大课听起来真的是挣扎..),作业要求用word文档,还有格式要求(但好像无所谓),我干脆用latex写出来了,贴在每章笔记末尾.

</font>

## 第一章 基本概念及定义
---
!!! note "知识梳理"  
    <font size = 3.4>
    
    🌟概念:热力系统与外界,开口系与闭口系,绝热系,孤立系,平衡状态,状态参数(T、p),表压力与真空度,可逆过程与准静态过程,温标
    
    🔧计算:压力计(表压$p_e$)与真空计(真空度$p_v$),温标相关计算

    </font>



## 第二章 能量与热力学第一定律
---



## 第三章 熵与热力学第二定律
---


## 第四章 热力学一般关系
---

!!! note "知识梳理"
    <font size = 3.4>

    🌟概念:简单可压缩系统的五个基本状态参数(p,v,T,u,s), 三个可测参数的状态方程$F(p, v, T)=0$, <B>热力学一般关系</B>,组合状态参数(h, f, g)

    🔧计算:

    </font>

>简单可压缩系统的特点:存在两个独立的状态参数,其状态函数为二元函数


根据热力学第一定律与第二定律, <B>简单可压缩工质在可逆变化中的能量平衡($F(u, v, s)=0$的全微分形式)</B>有:
     
\[
        \begin{aligned}
        &du = Tds - pdv                   \\
        \small 即:& \small热力学能 = 吸热量 - 做功量         \\                             
        &dh = -Tds - vdp (引入自由能f = u - Ts)                                    \\
        &dg = -sdT +vdp  (引入自由焓g = h - Ts)                                    \\
        \end{aligned}
\]

即通过Legendre变换,可得$F(h,s,p)=0$, $F(f,T,v)=0$, $F(g,T,p)=0$的全微分表达式.
再对上述的等式做一阶偏微商,有:

\[
        \begin{aligned}
        &( \frac{\partial u}{\partial s} )_v = (\frac{\partial h}{\partial s})_p = T                 \\
        -&(\frac{\partial u}{\partial v} )_s = -(\frac{\partial f}{\partial v})_T = p                                   \\
        &( \frac{\partial h}{\partial p} )_s = (\frac{\partial g}{\partial p})_T = v                 \\
        -&(\frac{\partial f}{\partial T} )_v = -(\frac{\partial g}{\partial T})_p = s                                   \\
        \end{aligned}
\]

由此可知, 对于$F(h,s,p)=0$, $F(f,T,v)=0$, $F(g,T,p)=0$,只需要知道任意一个关系式就能得到所有的状态函数

>将不可测的熵s与可测的p,v,T相联系

如上述的某些偏微商,具有明确的物理意义,将这些特殊的偏微商定义为<B>热系数</B>:

- 工质在定压条件下(因为p同样会影响体积,通过定压条件排除)的<B>热膨胀/体膨胀系数系数$[K^{-1}]$</B>: 

\[
        \alpha_{v} = \frac{1}{v} (\frac{\partial v}{\partial T})_{p}
\]

- 工质在等温条件下的<B>等温压缩率</B>$[Pa^{-1}]$:

\[
        \kappa_T  = - \frac{1}{v} (\frac{\partial v}{\partial p})_T
\]



待续未完.......😞