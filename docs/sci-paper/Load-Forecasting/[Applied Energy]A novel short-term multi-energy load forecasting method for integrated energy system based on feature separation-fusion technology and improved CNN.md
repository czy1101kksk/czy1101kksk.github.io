# [Applied Energy]A novel short-term multi-energy load forecasting method for integrated energy system based on feature separation-fusion technology and improved CNN

!!! info "📜Information"
    - 文章题目：*基于特征分离融合技术与改进CNN的综合能源系统多能负荷预测* 
    - Key word: `Integrated energy system` `Deep learning` `Multi-energy load forecasting` `Multi-task learning` `Convolutional neural network`
    - 作者：Ke Li, Yuchen Mu, Fan Yang, Haiyang Wang, Yi Yan, Chenghui Zhang

## 📢简述

---

>针对<B>IES数据量大、随机性强和多能耦合</B>的特点，文章中提出一种基于特征分离-融合技术与改进CNN的多能负荷短期预测方法。首先，基于静态图片像素点的分布规律将无明显规律的多能负荷数据点进行像素重构，使之在水平和竖直两个方向分别具有一定的关联特征。其次，采用特征分离—融合技术，基于信息价值差异，将输入特征分为两类进行差异化处理，并利用基于多尺度融合的改进CNN对重构后的三维负荷像素在高维空间内进行多尺度特征提取和融合。最后，将两类特征合并输入到以BiLSTM为共享层的多任务学习框架中，采用硬参数共享机制学习IES多能耦合信息。此外，为兼顾各种负荷不同的预测需求，设计了三种不同结构的FCN网络作为特征解释模块。实际算例表明，所提模型冬季日加权平均精度达98.01%，预测结果平均相对误差标准差低至0.0242，在所有对比模型中，预测精度最高，误差分布最稳定。

## 🎇Highlights

---

• Propose a feature processing method based on information value differences.

• Propose an improved CNN based on multi time scale fusion and reconstruct the original load data into a three-dimensional pixel matrix.

• Build a multi task hard shared learning framework based on BiLSTM, innovatively adopting feature interpretation modules with different structures.

• The simulation results show that the winter daily WMA of the proposed model can reach 98.01%, and the RESD is as low as 0.0242.

## 🎃文章观点

---

1.IES中多能量负荷预测的研究仍处于早期阶段。多能量负荷预测的应用场景复杂许多，需要同时考虑IES的外部因素和内部多能量流的相互传递和交叉耦合。而单负荷预测方法的扩展难以有效地学习多能量耦合信息。