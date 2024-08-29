[TOC]

# 文献标题

Title：A Detect-and-Verify Paradigm(范式) for Low-Shot Counting

- 影响因子：
- 期刊：
- 发表时间：2024年4月25日
- 阅读日期：2024年8月24日

> <font color='FF0000'>**TimeLine**</font>
>
> 2024年8月25日

## 第一遍：

> Background：（为什么研究这个？研究的target是什么？）

不仅counting 还要给出检测对象的 location和size

> Key words（记录一些设计思路的重点）

detection verify

> 随手记：（可能是你第一次了解到的概念 或者突然看到觉得有问题 有意思的地方）





## 第二遍：

> 实验原理/设计思路：（自己对整篇文章设计思路的理解）

> 实验内容（记录一些它在results里具体做了什么）

> Advantages（文章的优点，思考为什么人家能打高分？）

> Disadvantages（这篇文章有什么不足的地方吗？不强求，但批判性思维）

> Enlightenment（对自己课题的启发？有什么实验方法可以借鉴？实验思路可以学习？）







# Abstract

> **研究现状（最新的方法在当前领域取得了什么样的成功，克服了什么问题）**

- 少样本计数器 标记框&图像  
- 主流的计数方法  密度图中对象定位的密度之和

> 存在问题：但是仍存在什么问题没有解决

- 不提供目标的定位和尺寸→基于检测的计数器→准确率降低

- 假阳性很高

> **提出解决方法（模型）：提出的创新模块&模型结构（简要概述其核心创新点），**
>
> **能够解决该问题的理论支撑是什么？**

- DAVE  基于检测和验证范式
- 第一阶段：高召回率的检测集合
- 第二阶段：验证检测，识别并移除异常值

> 意义、结果，达到了怎样的实验效果

1. 基于密度 MAE ~20%
2. 基于检测 检测质量 ~20%
3. 零样本 & 基于文本提示 这两个技术领域 都达到了sota

# Introduction

> - 研究领域：其领域具体的运用场景和重要性
> - 研究历史：①早期的方法、模型以及存在的问题  ②近期方法和模型以及优势
> - 存在问题：近期方法也还是存在的问题 或者 是长期的共性问题
> - 如何解决：本文提出的模块 和 模型结构如何解决该问题
> - 贡献：① 创新模块 ② 模型结构 ③实验证明其sota
>

P1：样例框、少样本、基于密度、基于检测

P2，S1：基于密度，不提供位置和尺寸（这个问题有多么的严重，有待解决

P3，S1：基于密度 & 基于检测 → 多目标类别的情况下

> 【问题】高召回率：泛化选定对象的各种appearance→FP假正例→P 🔽 countings 🔼（高估）
>
> 解决：训练图像选择多类别 $\rightarrow$ R $\downarrow$ countings $\downarrow$

P4

> S1：LOCA可以解决上面的问题：①提供location size ②保证召回率和准确率
>
> ​	基于检测的：location size  PR 低
>
> ​	基于回归的：PR   无location size 
>
> ​	∴基于检测和验证的范式
>
> S2：提高现有计数器针对特定类别的泛化能力
>
> S3：检测阶段（应用基于密度的优点 生成一个高召回率的候选集）
>
> S4：验证阶段：提高准确率（识别并丢弃outliers）
>
> S5：对应在的检测阶段的密度图region也被删除
>
> S6：DAVE 是第一个即适用zero-shot & text prompt的计数器

P5：

> 1. 贡献：LOCA 高召回率 高准确率
>
> 2. 适用于所有的少样本计数场景
>
> 3. 既有基于密度的优势 又有 就有检测的优势 还是第一个 零样本计数器 输出基于检测的结果
>
> 4. ==结果：在所有基于密度==的计数方法中: DAVE $\rightarrow$SOTA  减少了）<u>20%（MAE）和43%（RMSE）</u>
>
> 5. ==结果：就算是在所有基于检测==的计数方法中，依然是SOTA（仅限于FSCD147数据集）（检测指标提高了**20%**、总技术估计提高**38%**）
>
> 6. 在基于文本提示的计数领域：新的标杆
>
> 7. 变体DAVE（零样本的）：
>
>    ①比所有基于密度的==零样本==计数任务：outperform（优于）
>
>    ②on-par（持平）：检测准确率 // 在最近的==少样本==计数器上持平
>
> 8. ==计数的设置==可以改，但DAVE 就是outperforms（不管是基于密度的还是基于检测的计数器）

ME：通过4个数字说明DAVE的性能

# Related Work

> - 早期方法概述：总结列举模型名 & 可能面临的问题  
>

**P1**

对特定目标的检测和目标计数是一起出现的，比如交通工具、细胞

因为密集场景计数不佳，所以出现了基于密度的计数方法。

**P2**

为了训练针对特定类别的技术模型，都需要大量的标注数据集，但实际上哪有那么多数据集

**P3**

1. 所以出现了类别不敏感方法，就是在测试集上测试时能够适应很多目标类别，同时使用最小的监督信号
2. 早期的研究：使用孪生匹配网络预测密度图
3. FSC147数据集的提出促进了 少样本计数的发展
   1. Famnet 在测试阶段对骨干网络进行调整 $\rightarrow$ 提高密度图的估计
   2. BMNet+ 提高定位 、减少类内变异性
4. SafeCount 提高类别泛化能力
5. CountTR
   1. 图像特征提取 Vit
   2. 样例特征提取 卷积编码器
   3. 特征交互模块 交叉注意力  交互什么？ 图像特征和样例特征
6. LOCA  提出了OPE 通过迭代适应融合外观特征和形状特征

> - 近期方法详述：列举模型名 并一句话概括其特点

P4：key words：with the recent development ...

1. 就是说所有的少样本图像计数问题都需要 样例框 来指定类别
2. 基于文本提示的计数模型出现了
3. 不需要样例框标注，只要有对目标类别的描述就行，下面开始罗列方法
   1. ZeroCLIP 选择图像中的Patches 作为示例 
   2. CLIPCount  ① 图像文本对齐 ② 对比损失

4. 还有一些工作比如 零样本计数 是在不提供示例框的情况下 对主要类别进行计数

P5：

微小结构的改变，使得少样本计数方法 也能应用到 零样本计数

基于密度的最大缺点就是不提供目标定位

P6：

为了解决基于密度的计数没有目标定位的问题，已经有工作了。但是吧，计数不准。

> - 本模型概述：如何比近期方法更好的去解决问题

基于检测和验证

DAVE：①少样本计数 ②检测方法  基于检测和验证

**检测阶段：**

​	保证高召回率、生成很多候选框

**验证阶段**

​	识别和删除 异常值 以提高准确率

​	异常值可以用来 ①更新密度图 ②提高基于密度的计数估计

​	在这个阶段 有可能会补充遗漏的计数对象 或者 和检测数量 $N_P$ 不同

![image-20240825193948733](../../../Users/dearr/Library/Application Support/typora-user-images/image-20240825193948733.png)

# Method

> - 数学表示法明确
>

输入图像：$I \in \mathbb{R}^{H_0 \times W_0 \times 3}$

k个真实示例框：示例框：$B^E = \{ b_i\}_{i=1:k}$

估计来的： $B^P = \{ b_i \}_{i=1:N_P}$

> - 创新模块：自身创新方法 对比 近期方法（一张图）

⭐️检测阶段【me ：从后往前的叙述网络 总分的结构】

🟢 P1 ： 怎么预测边界框？检测阶段的第1段说的就是这里了

<img src="../../../Users/dearr/Library/Application Support/typora-user-images/image-20240825211433274.png" alt="image-20240825211433274" style="zoom:50%;" />

1. 预测候选边界框  $B^C =\{b_i \}_{i=1:N_c}$ 

2. 如何预测候选框？

   ① 首先，预测中心    $C = \{ x_c^i,y_c^i\}_{\{ i= 1:N_C\}}$ 

   ② 预测相应的边界框参数【ME：都有什么边界框参数？】

   <img src="../../../Users/dearr/Library/Application Support/typora-user-images/image-20240825211508958.png" alt="image-20240825211508958" style="zoom:33%;" />

3. $\tilde{G}$ : 目标密度图  LOCA

   $C$ 非极大值抑制得到定位中心  non-maxima suppression

   <img src="../../../Users/dearr/Library/Application Support/typora-user-images/image-20240825211643115.png" alt="image-20240825211643115" style="zoom:50%;" />

   

4. $ \tilde{G}$ 的估计方法（就是简要概括了一下LOCA，记几个数学符号即可）

   <img src="../../../Users/dearr/Library/Application Support/typora-user-images/image-20240825211849238.png" alt="image-20240825211849238" style="zoom:50%;" />

   1.  图像的处理→ $f^I \in \mathbb{R}^{h \times w \times d}$   【怎么来的？是什么？原文都有】
   2.  样例框的处理：OPE模块→示例原型  【事实上 $f^E$也参与了原型提取 】
   3.  $f^I$ 和示例原型 做相关运算：  $\tilde{R} \in \mathbb{R}^{h \times w \times d}$
   4.  相似性张量 → 2D密度图的过程：$\tilde{R} \overset{\text{解码器}}{\rightarrow} \hat{G} \in \mathbb{H_0 \times W_0} $

【me】 $\tilde{G}$  (真的估计出来的密度图)  和 $\hat{G}$ （一系列估计出来的密度图）

🟢 P2：

第二段说的就是这里了    // 文字描述和符号都对上了 【先说的主干  开始说的分支】

<img src="../../../Users/dearr/Library/Application Support/typora-user-images/image-20240825210800783.png" alt="image-20240825210800783" style="zoom:50%;" />

预测示例框参数

根据检测中心 构造特征 预测 示例框参数【Q 其实我不懂，什么是示例框参数？】

怎么构造特征？

①特征构造的原则是 反应目标信息

② 具体怎么实施的：

<u>第一步</u>：从骨干网2 3 4 阶段抽取的特征被 resize $64 \times 64$ 

​	→ 将其沿着通道维进行拼接 

​	→ 使用 $3 \times 3$的卷积降到d通道 即 $f^0 \in R ^{h \times w \times d}$

​	【Q】为什么这么做是有效的？为什么这么做就叫构造特征了？

<u>第二步</u>： $ f^0$ 上采样到跟输入尺寸匹配 记成 $f^1 \in \mathbb{R}^{H_0 \times W_0 \times d}$

​	【Q】  $h$ =? 64  $H_0$  = ? 512

<u>第三步</u>：FFM  特征融合模块

形状信息在特征融合模块中还要注入。【Q:哪里注入了？图上也看不出来啊】

$\tilde{f} = FFM (f^1,\tilde{R} )$

🟢 P3：   从边界框回归头 怎么得到边界框候选集 $B^C$

$\tilde{f} \rightarrow \Omega(\cdot)$

- $\tilde{f}$ 哪里来的？看上文

-  $\Omega( \cdot)$ 边界框回归头 

  - 回归什么的？ ~预测边界框中心点 上下左右的边缘【这句话没看懂】

  - 结构是什么？

    ① 两个d通道的 $3 \times 3$的卷积层

    ②ReLu激活函数层

    ③4通道的 GroupNorm层

  - For what，最终的输出？预测的密集边界框映射  $v \in \mathbb{R}^{H_0 \times W_0 \times 4}$  【为什么是4？】
  - $v$ 在 $C$ 处 对应的值 得到 目标候选边界框 $B^C$  【就是说 具体地形状 代数都没. . .】

⭐️验证阶段

P1 注意符号和对应的含义

>1. 候选检测集 $B^C$ 高召回率 & 高误报率
>2. 检测阶段的主要目标就是提高准确率，通过 ① 分析外观 ② 删除异常值
>3.  从  $b_i$ 边界框 **抽取**  $f_i^V$验证特征向量 步骤如下：
>   1. backbone 的特征 记为 $f^0  \overset{池化}\rightarrow f_i \in \mathbb{R}^{s \times s \times d} \overset{\phi(\cdot)= 2个 1\times 1 conv(channel = d) 中间有 BatchNorm 和 ReLu} \rightarrow$
>   2. 验证阶段 示例框特征也要提取 共提取  $\N_C + k$ 个  记号：   $ F^V = \{ f_i^V\}_{i=1:(N_C+k)\cdot}$

P2

> 1. 对抽取到的验证特征进行聚类
> 2. 谱聚类，亲和度矩阵计算，余弦相似度， $F^V$ 特征对   ===》 由 $F^V$ 特征对之间的余弦相似度计算亲和度矩阵进行谱聚类 聚类成簇
> 3. 对象候选检测，属于某个簇的 有至少一个样例框被保留 =》 那些候选框属于某个簇，簇里至少有一个样例框，那么这个候选框被保留，其余的的则被标为异常点，被移除。最后产生 $N_P$个目标检测集合   =》记为： $B^P =\{b_i\}_{i=1:N_P}$
> 4.  更新$\hat{G} \rightarrow G$  方法是 把检测边界框以外的值设置为 0===》更新完以后得到密度估计

<img src="../../../Users/dearr/Library/Application Support/typora-user-images/image-20240826185822744.png" alt="image-20240826185822744" style="zoom:50%;" />

⭐️ 零样本计数和基于提示的适应【这里是在说 DAVE的泛化场景

🟢 Zero shot counting.

首先给出零样本计数的定义：不给样例框，检测并计数主要类别

接下来给出具体的实施过程：首先密度图的预测使用 LOCA零样本的变体，其次检测阶段不变，验证阶段稍微改变。  

接下来给出验证阶段怎么做的改变：验证阶段的改变主要在于聚簇的选择: 保留 那些 所有 大小为 最大簇的45%（至少），其余的当做异常值。

Prompt-based counting.

【原理是什么？】适用于零样本情形下的DAVE模型，可以拓展到基于提示的计数，只是要计数的目标由文字指定

唯一的修改就是验证阶段簇的选择方法

文字嵌入特征由clip提取，然后文字嵌入特征与 每一个簇的嵌入特征进行比较，这个嵌入特征也是由clip生成

【接下来说簇嵌入怎么来的】图像掩码、边界框以外 计算clip嵌入

【异常值判别原则】计算 文本嵌入 和独立簇嵌入 的余弦相似度，少于最高相似度的85%被判断为异常值。

> - 创新结构：由哪几个部分组成（整体流程图）

3.4 training

P1

> 1. 【少样本计数数据集的数据格式】少样本计数数据集 所有对象标记的中心 $k=3$个样例框
> 2. 【训练时 注意数据集的格式】训练的原则是 数据集的格式
> 3. 【对象中心主要用来密度的定义预测】对象中心 训练 密度图定位网
> 4. 【关于训练参数的说明】DAVE 使用LOCA 做初始密度估计 LOCA 使用公开的预训练权重  DAVE训练的检测和验证阶段的自由参数

P2

>【检测阶段的训练细节 详细说明了损失函数】检测阶段 =  FFM +  $\Omega(\cdot)$ 
>
>检测阶段的训练过程 由边界框损失监督 边界框损失是评估有效的真实样例框，即，
>
>$\mathcal{L} = \sum_{i=1}^{k=3} 1- \text{GIoU}(\mathbf{v}({x^c,y^c}),b_i^{ \text{GT}})$ 
>
>【meaning】
>
>$(x_c^{(i)},y_c^{(i)})$ 真实样例框 $b_i^{\text{GT}}$  的中心点
>
>$\text{GIoU}(\cdot)$ ：广义交并比

P3 【详细说明验证阶段的训练过程】

> 首先，验证阶段的特征抽取网络 记为 $\phi(\cdot)$
>
> 【训练样例如何生成？】把不同类别有样例标注的图像对 缝合起来
>
> 【缝合图像的一些性质】有6个边界框、由 $ \phi(\cdot) $抽取 两个特征，对应两个样例集合 
>
> ​	$ \{z_j^1\}_{j=1:3}$	$ \{z_j^2\}_{j=1:3}$	
>
> 【验证阶段的网络如何训练？】  训练阶段的网络 是 $\phi(\cdot)$
>
> ​	由对比损失训练
>
> ​		<img src="../../../Users/dearr/Library/Application Support/typora-user-images/image-20240827133727473.png" alt="image-20240827133727473" style="zoom:25%;" />
>
> 【meaning】 $c(z_{j_1}^{i_1},z_{j_2}^{i_2})$特征对之间的余弦相似度   $\lambda$ 是间隔

# Experiment

> - 介绍运用的数据集
>

【预处理 & 训练中的实验细节】

> - 对比实验：模型结构、模块（表格）



> - 消融实验：模块（表格）

Impact of mixed-class training

> - 泛化实验：任务、数据集

1. 基于密度的计数 性能表现

   1. 数据集 FSC147
   2. Protocol MSE  RMSE
   3. Few_shot Counting
      1. 【和谁比？】
      2. 【结果 总结】outperform  table1
      3. 【原因：讨论为什么好】
   4. one-shot counting
   5. prompt-based counting     $\mathrm{DAVE_{prm}}$
   6. zero-shot counting     $\mathrm{DAVE_{0-shot}}$

   【我有话】针对一个基于密度的计数任务下，刷了 4个场景 1个数据集 两个指标 若干模型

2. 检测 性能表现

   1. Few-shot dectection ： 

      1. 数据集 ： FSCD147
      2. protocol：①AP  ② AP50
      3. 【和谁比？】
      4. 【又刷2 个数据集】：$\text{FSCD-LVIS}$、$\mathrm{FSCD-LVIS_{uns}}$

      【summary】 这个场景下 刷了 3个数据集、2个指标、6个方法的比较，表格的画法：

      <img src="../../../Users/dearr/Library/Application Support/typora-user-images/image-20240827135824404.png" alt="image-20240827135824404" style="zoom:25%;" />

   2. zero-shot detection

      1. 【数据集】 没说

      2. 【和谁比】Count&detection 只有 $\mathrm{DAVE_{0-shot}}$  

         ​		还有 $ \mathrm{ C-DETR}$ 但是输入需要三个样例框

      3. 【结果】尽管如此 
      
         1. 【AP50】**验证集**上还是outperform  指标是AP50 **测试集**上性能表现相当  ===》 检测的鲁棒性相当
         2. 【AP】定位准确性（AP meaning）较低
      
      4. 【summary】尽管如此吧 DAVE毕竟是不需要示例框的 实打实的 zero-shot
      
   
   	3. Few-shot detection counting
   
       	1. 【再刷一个场景  检测+计数】少样本检测计数  = 检测框之和（与密度检测区分）
       	2. 【和谁比？ $ \mathrm{DAVE^{box}}$ 】
       	3. 【数据集】
       	4. 【protocol】
       	5. 【结果 & 总结】
   
       <img src="../../../Users/dearr/Library/Application Support/typora-user-images/image-20240827144319596.png" alt="image-20240827144319596" style="zoom:25%;" />
   
       <img src="../../../Users/dearr/Library/Application Support/typora-user-images/image-20240827144128885.png" alt="image-20240827144128885" style="zoom:25%;" />
   
   【summary】计数性能 做了 检测性能 做了 检测性能+计数性能 也做了

> - 实验总结

结合表1和表8   下结论：

 not only outperforms all detection-based counters, but also all published density-based counters in terms of MAE

# Conclusion

> - 提出存在的问题
>



> - 受到的启发如何解决



> - 提出方法如何解决



> - 未来工作



