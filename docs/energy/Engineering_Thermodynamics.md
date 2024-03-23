# 🍥工程热力学(甲)
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
???+ summary "🌇Information"
    <font size =4>
    
    - 课程:工程热力学(甲) Engineering Thermodynamics(A)
    
    - 学分:4.0             课程代码: 59120030
    
    - 教师:俞自涛老师       教材:<工程热力学(第三版)>(曾丹苓、敖越等) 
    
</font>
<font size = 3.5>

>虽然不是慧能班的,但在无意中选到了慧能班的课,自己和俞老师还算比较熟悉,班上一共只有20多人所以听课体验还不错(某些百人的大课听起来真的是挣扎..),作业要求用word文档,还有格式要求(但好像无所谓),我干脆用latex写出来了,贴在每章笔记末尾.

</font>

## 第一章 基本概念及定义
---
!!! note "知识梳理"  
    <font size = 3.4>
    
    🌟概念:热力系统与外界,开口系与闭口系,绝热系,孤立系,平衡状态,状态参数(T、p),表压力与真空度,可逆过程与准静态过程,温标
    
    🔧计算:压力计(表压$p_e$)与真空计(真空度$p_v$),不同温标之间的相关计算

    </font>

### 热力系统

热力系统是人为分割出来的作为热力学分析对象的<B>有限物质系统</B>,而与系统进行<B>质能交换</B>的物体为外界:

- 闭口系:与外界只有能量交换而没有物质交换(<B>物质不透过边界</B>),即控制质量$\frac{dm}{dt}=0$($C.M.$)

- 开口系:不仅有能量交换也有物质交换,即控制体积/控制体($C.V.$).
> 开口系与闭口系的区分在于<B>有没有质量越过边界</B>

- 绝热系:系统与外界没有热量交换

- 孤立系:既没有能量交换也没有物质交换(显然,孤立系必然为绝热系)
>孤立系不代表不发生作用,而是一切相互作用都发生在系统内部,如:将某一系统以及与之发生质能交换的外界看作一个大的系统,该联合系统即为孤立系统

- 可压缩系统:由可压缩流体构成的热力系统

- 简单可压缩系统:仅有<B>准静态体积变化功</B>(膨胀功、压缩功)的可压缩系统

<font size = 4>状态参数→</font>描述工质所处<B>平衡状态</B>的宏观物理量,反映的是大量分子运动的宏观平均效果(由此可知,状态方程是在平衡状态下,状态参数的关系式)

!!! note "平衡状态"
	<font size = 3>
	在不受外界影响的条件下(系统与外界的不平衡势消失)系统的内部状态能始终保持不变(宏观变化全部停止,宏观性质不变)即为<B>平衡状态</B>

	><B>只有平衡状态的系统才能用状态参数来描述,只要有两个独立状态参数即可确定一个状态,所有其他状态参数均可表示为这两个状态参数的函数</B>


	</font>

>当热力系经历一封闭状态变化过程而又恢复到原始状态时,其状态参数的变化为0,即:$\oint d \xi= 0$

- 基本状态参数:p, V, T

- 强度量(p, T):与物质的数量无关,不具有可加性,对<B>处于平衡状态的系统</B>才具有确定的数值

- 广延量/尺度量(V, U, H, S):与工质质量成正比,具有可加性

???+ advice "温度"
    <font size = 3.4>
    温度是描述和判断系统与其他系统/外界处于<B>热平衡状态</B>的参数
    
    !!! note  "热力学温标(绝对温标)"
        热力系温标(K)将水的三相点温度定为基准点,规定为273.16K,而绝对零度为0K,而其他温标与热力系温标仅是零点取值的不同.
    
    <B>温标转换</B>:通过将新温标(线性)与已知温标(如热力学绝对温标)的尺度作对应,求得两种温标的线性关系,设某一温标($^oN$)在1个标准大气压下的冰点与汽点为$T_1^oN与T_2^oN$,已知热力学绝对温标(K),则
	
	\[	
		\frac{T_2 - T_1}{373.15 - 273.15} = \frac{ \{T_{N}\}_{^oN} - T_1 }{ \{T_{N}\}_{K} - 273.15 }          
	\]
	
	由此,即可求得两个温标之间的线性关系: $\{T_{N}\}_{^oN} = \frac{T_2 - T_1}{373.15 - 273.15} (\{T_{N}\}_{K} - 273.15)$
    
	</font>

!!! note "压力"
    <font size =3.8>
	
	<B>单位面积</B>上所受的垂直作用力为压力(压强)[$Pa$],测量工质压力的仪器为压力计

	!!! note "压力计的工作原理"
		<font size =3.4>
		
		压力计测量的是工质绝对压力$p$与外界环境压力$p_b$之差,即相对压力/表压($p_e, p_v$),而绝对压力与大气压力无关,因此大气压力变化并不会影响绝对压力:

		\[
			\begin{equation}
			p=
			\begin{cases}
			p_b + p_e &, \text{ $ p > p_b $ } (p_e为表压力)   \\
			p_b - p_v &, \text{ $ p < p_b $ } (p_v为真空度)   \\
			\end{cases}
			\end{equation}
		\]

		</font>

		![](img/pe-pv-p.jpg)

    </font>    

### 准静态过程
---
状态变化无限缓慢,弛豫时间很短而<B>无限接近平衡状态</B>的过程
><font size = 4><B>准静态条件</B>:气体工质与外界之间的温差/压力差为无限小,若还存在其他不平衡势则必须加上相应的无限小条件,<B>只有准静态过程在坐标图中可用连续曲线来表示</B></font>
	
\[
	即: p \rightarrow p_{out} + \frac{F}{A}, T \rightarrow T_{out}
\]

### 可逆过程: 准静态 + 无耗散:
---
准静态过程中可能会发生能量的耗散(比如外部机械器件之间摩擦力的作用),而可逆过程着眼于工质与外界作用产生的总效果,<B>不仅要求工质内部是平衡的,而且工质与外界可以无条件的逆复,该理想过程中不存在任何耗散(即返回原来状态并且在外界不留下任何变化)</B>

!!! note "过程热量的概念"
	"热量"[$J$]一词为热力系与外界之间仅仅由于温度不同而通过边界传递的能量,是用来度量传递能量多少的<B>过程量</B>,与状态参数(只取决于初、终态)不同,过程量与进行的路径有关,不能表示为状态参数的函数

	在<B>可逆过程</B>中,热量可表示为:
	
	\[
		\Delta q = Tds, q_{1-2} = \int _{1}^{2} Tds
	\]

	![示热图](img/heat.jpg){width=300px height=200px}


!!! info "可逆状态下的功"
	
	<font size = 3.4>
	>功[$J$]与热量一样是能量传递的度量,是过程量,,不能表示为状态参数的函数($w \neq f(p,v)$),热力学规定对外做功为"+",外界对系统做功为"-"

	对于可逆过程中的体积变化功:

	\[
		\Delta W = Fdx = pAdx = pdV, W_{1-2} = \int _{1}^{2} pdV
	\]

	![示功图](img/w.jpg){width=300px height=200px}
	</font>

	!!! note "有用功"
		
		<font size = 3.4>
		闭口系工质膨胀所做的功并不全部有用于膨胀,有一部分用来排斥大气和摩擦耗散,余下的才是有用功$W_{u}$,即

		\[
			W_u = W - p_{0} \Delta V - W_f
		\]

		若为可逆过程,则$W_f=0$,可得:

		\[
			W_u = \int _{1}^{2} pdV - p_{0} \Delta V 
		\]

		</font>

### 第一章作业

<object data="../home-work/Engineering-Thermodynamics-homework-1.pdf" type="application/pdf" width="100%" height="800">
	<embed src="../home-work/Engineering-Thermodynamics-homework-1.pdf" type="application/pdf" />
</object>

## 第二章 能量与热力学第一定律
---
!!! note "知识梳理"  
    <font size = 3.4>
    
    🌟概念:热力系第一定律,热力学能$U$,焓$H$的定义,推进功和流动功,开口系能量方程,稳定流动能量方程
    
    🔧计算:开口系能量方程,稳定流动能量方程

    </font>

!!! note "推进功和流动功"

	<font size = 4>
	推进功差$p_2 v_2 - p_1 v_1 = \Delta (pv)$是维系工质流动所需的功,称为流动功

	<B>流动功</B>可视为流动过程中系统与外界由于物质进出而传递的机械功
	![推进功](img/pv.jpg)
	
	</font>

!!! note "焓的定义"

	<font size = 4>
	焓[$J$]是系统中因引入/排除工质还改变的总能量(热力学能+推进功),即$H=U+pV$

	显然,焓是一个状态参数,有$\Delta h_{1-2} = \int_{1}^{2} dh = h_2 - h_1$, $\oint dh = 0$
	</font>

### 热力学第一定律的表达式
---

根据热力学第一定律的原则,<B>系统中储存能量的增加 = 进入系统的能量 - 离开系统的能量</B>,对一个微元过程,有:

\[
	\delta Q = dU + \delta W
\]

对于可逆过程,有:

\[
	\delta Q = dU + pdV, Q = \Delta U + \int_{1}^{2}pdV
\]

对于循环,$\oint dU = 0$,有:

\[
	\oint \delta Q = \oint dU + \oint \delta W = \oint \delta W , 即Q_{net} = W_{net}
\]

可知在循环中,交换的净热量$Q_{net}$等于净功量$W_{net}$

### 开口系能量方程
---
如下图的开口系场景,在$d \tau$时间内的微元过程,$\delta m_1,dV_1 \rightarrow \delta m_2,dV_2$,系统从外界接收热量$\delta Q$,工质对机器设备做功$\delta W_i$(内部功),系统总能量增加$dE_{cv}$
![开口系能量传递](img/outdoor.jpg)

\[
	\begin{aligned}
	\delta Q &= dE_{cv} + (dE_2 + p_2 d V_2) - (dE_1 + p_1 d V_1) + \delta W_i   \\
	&\Rightarrow  dE_{cv} + (\frac{1}{2}c_{2}^{2} + gz_2) \delta m_2 - (\frac{1}{2}c_{1}^{2} + gz_1)\delta m_1 + \delta W_i    \;   (*)  \\
	\end{aligned}
\]


### 稳定流动能量方程
---
对于稳定流动,热力系任何截面上工质的所有参数都不随时间改变,因此必要条件为$\frac{dE_{cv}}{d \tau} = 0$, 因此上述(*)式变为:

\[
	\begin{aligned}
	\delta Q &= dH + \frac{1}{2} m dc^2 + mgdz + \delta W_i  \\
	\delta q &= dh + \frac{1}{2} dc^2 + gdz + \delta w_i  \\
	\end{aligned}
\]

其中$\delta w_i$代表1kg工质进入系统后在机器内部做的功

实际上将等式后三项的机械功称为技术功$W_t = \frac{1}{2} m dc^2 + mgdz + \delta W_i$,因此:

\[
	\begin{aligned}
	Q &= \Delta H + W_t  \\
	\delta Q &= dH + \delta W_t  \\
	\delta q &= dh + \delta w_t  \\
	\end{aligned}
\]

对于上述公式还可以继续变形,引入容积变化功:

\[
	\begin{aligned}
	W_{\text{容积变化功}} &= q - \Delta u =  \frac{1}{2} \Delta c^2 + g \Delta z + \Delta (pv)  \\
	&\Rightarrow W_t + \Delta (pv)         \\ 
	W_t &= W_{\text{容积变化功}} - \Delta (pv)  \\
	\text{对于可逆过程:} W_t &= \int_{1}^{2}pdV - \Delta (pv)  \\
	&= \int_{1}^{2}pdV - \int_{1}^{2}d(pv) = -\int_{1}^{2}vdp  \\
	 即为 \delta w_t &= -vdp  \\
	 \delta q &= dh + \delta w_t = dh -vdp  \\
	 \delta Q &= dH -Vdp  \\
	\end{aligned}
\]

## 第三章 熵与热力学第二定律
---
!!! note "知识梳理"  
    <font size = 3.4>
    
    🌟概念:热力学第二定律的两种表述,
    
    🔧计算:卡诺定理

    </font>

!!! info "热力学第二定律"

	<font size = 3.4>
	
	- 克劳修斯表述(热量传递角度):不可能将热从低温物体传至高温物体而不引起其它变化。
	
	>强调"自发地,不付代价地"
	
	- 开尔文表述(热功转换角度):不可能从单一热源取热，并使之完全转变为有用功而不产生其它影响。

	>不可能将从热源取得的热全部转换为功,不可避免地将一部分传递给温度更低的低温热源

	<B>非自发过程(热转为功/低温向高温传热)的实现必须有一个自发反应(高温向低温传热/机械能转变为热能)作为补充条件</B>
	
	</font>

### 卡诺定理
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