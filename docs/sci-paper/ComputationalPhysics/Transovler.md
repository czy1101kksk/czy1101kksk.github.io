# Transolver: A Fast Transformer Solver for PDEs on General Geometries

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>

!!! info "相关信息"
    <font size = 3.5>
    
    论文地址：[Transolver: A Fast Transformer Solver for PDEs on General Geometries](https://arxiv.org/pdf/2402.02366v2)

    代码（Pytorch版）:[https://github.com/thuml/transolver](https://github.com/thuml/transolver)

    </font>

### Abstract

Since PDEs are typically discretized into large-scale meshes with complex geometries, it's hard for traditional Transformers to capture intricate physical corrlations directly form massive individual points.

So the team present ```Transolver(Physics Attention)``` learning intrinsic physical states hidden behind discretized geometries.```Physics Attention``` adaptively split the discretized domain into a set of slides with flexible shapes.by calculatiing attention to physics-aware tokens encoded from slices, ```Transolver``` can effectivly capture the physical correlations under complex geometries and compute in linear complexity.

### Method 

<B>Problem Defination</B>:

Input domain $\Omega \subset \mathbb{R}^{C_g}$, where $C_g$ denotes the dim of input space.$\Omega$ is discretized into a set of meshes points $\mathbb{g} \subset \mathbb{R}^{C_g \times N}$.The task is to estimate target physical quantities based on input geometrics $\mathbb{g}$ and quantities $\mathbb{u} \in \mathbb{R}^{N \times C_u}$.

The key to solving PDEs is to capture intricate physical correlations.But it's hard for ```Attention ``` to learn from such massive discretized mesh points.