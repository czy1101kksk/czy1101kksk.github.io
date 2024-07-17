# <B>Rohlik Orders Forecasting Challenge</B>
---
> Use historical data to predict customer orders
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>

!!! info "概述"
    
    <font size = 3>
    Rohlik Group，这个领先的欧洲电子产品创新企业，正在彻底改变食品零售业。我们在捷克共和国、德国、奥地利、匈牙利、罗马尼亚等地设有11个仓库。

    我们的比赛聚焦于预测未来60天的选定仓库的订单（杂货配送）数量。
    
    </font>

评测得分：

- 提交的订单根据预测订单和实际订单之间的平均绝对百分比误差进行评估。

提交文件：

- 对于测试集中的每个ID，您必须预测订单的数量。该文件必须包含一个标题，并具有以下格式：

```
ID,ORDERS
Prague_1_2024-03-16,5000
Prague_1_2024-03-17,5000
Prague_1_2024-03-18,5000
etc.
```

!!! info "Timeline"

    - 2024年8月9日：首次提交截止日期。你的团队必须在截止日期前提交第一份文件。
    
    - 2024年8月9日：团队合并截止日期。这是您可以与另一个团队合并的最后一天。
    
    - 2024年8月23日：最终提交截止日期

<B>Dataset Description</B>:

你将被提供选定的Rohlik仓库的历史订单数据，任务则是预测测试集中的“订单”列。其中一些特征数据在测试中不可用，因为在预测时尚不清楚（例如降水、停机、网站上的用户活动）。

- ```train.csv```-包含历史订单数据和下文所述选定特征的训练集

- ```test.csv```-测试集

- ```solution.example.csv```-格式正确的示例提交文件

- ```train_calendar.csv```-训练集的日历，包含有关假期或仓库特定事件的数据，有些列已经在训练数据中，但此文件中还有其他行，用于显示某些仓库可能因公共假期或周日而关闭的日期（因此它们不在训练集中）

- ```test_calendar.csv```-测试集的日历

!!! advice "```train.csv```"

    <font size = 3>
    - ```warehouse```: 仓库名

    - ```date```: 日期

    - ```orders```: 归属于该仓库的客户订单数量

    - ```holiday_name```: 公共假日名称（如有）

    - ```holiday```: 0/1 表示有假期

    - ```shutdown```: 仓库因操作而停机或受限（测试中未提供）

    - ```mini_shutdown```: 仓库因操作而停机或受限（测试中未提供）

    - ```shops_closed```: 公共假日，大多数商店或大部分商店关闭

    - ```winter_school_holidays```: 学校假期

    - ```blackout```: 仓库因操作而关闭或限制（测试中未提供）

    - ```mov_change```: 最小订单值的变化，表明客户行为的潜在变化（测试中未提供）

    - ```frankfurt_shutdown```:仓库因操作而停工或限制（测试中未提供）

    - ```precipitation```: 仓库位置周围与客户位置相关的降水量（mm）（测试中未提供）

    - ```snow```:仓库位置周围与客户位置相关的降雪量（mm）（测试中未提供）

    - ```user_activity_1```:网站上的用户活动（测试中未提供）

    - ```user_activity_2```:网站上的用户活动（测试中未提供）

    - ```id```: 由仓库名称和日期组成的行id

```train_calendar.csv```:

包含 train.csv 列，但包含更多行，因为它包含所有日期，而 train.csv 不包含仓库因公共假期或其他事件而关闭的日期



    </font>