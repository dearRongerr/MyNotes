# 学习框架

- 是什么（原理
- 为什么（bg 要解决的问题
- 怎么做（代码
- 是谁（作者
- 在哪儿（期刊
- 啥时候（时间
- 优点（解决了过去的什么问题
- 缺点（未来的研究趋势

# CNN变体

> 思想很重要

- LeNet
- AlexNet
- VGG
- NiN
- GoogLeNet
- ResNet
- DenseNet

# 1 LeNet

- 是什么（原理

<img src="/Users/dearr/Library/Application Support/typora-user-images/image-20240305134936065.png" alt="image-20240305134936065" style="zoom:50%;" />

![image-20240305135108472](/Users/dearr/Library/Application Support/typora-user-images/image-20240305135108472.png)

<img src="/Users/dearr/Library/Application Support/typora-user-images/image-20240305135317396.png" alt="image-20240305135317396" style="zoom:50%;" />

<img src="/Users/dearr/Library/Application Support/typora-user-images/image-20240305154952357.png" alt="image-20240305154952357" style="zoom:50%;" />

**通常我们说的填充是指在一边填充多少！！！！！！！！！！！！！！！！！！！**

只有书上，故意的搞出 $p_h、p_w$迷惑人心

原文的网络架构：

<img src="/Users/dearr/Library/Application Support/typora-user-images/image-20240305161217683.png" alt="image-20240305161217683" style="zoom:50%;" />



- 为什么（bg、要解决的问题、为什么网络架构这么设计？为什么是这样的卷积核参数？

  <img src="/Users/dearr/Library/Application Support/typora-user-images/image-20240305155146219.png" alt="image-20240305155146219" style="zoom:50%;" />

  Ps：这是一个伟大的思想，在于这是第一个从全连接网到卷积网的实现，能想到这种思想的作者真的很棒。我认为我们不仅要学习这个算法的是什么，更应该思考么作者能从全连接想到卷积核这种参数共享的方法

- 怎么做（代码

```python
net = nn.Sequential(
    nn.Conv2d(1,6,kernel_size=5,padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2,stride=2),
    nn.Conv2d(6,16,kernel_size=5),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2,stride=2),
    nn.Flatten(),
    nn.Linear(16*5*5,120),
    nn.Sigmoid(),
    nn.Linear(120,84),
    nn.Sigmoid(),
    nn.Linear(84,10)
)
X= torch.rand(size=(1,1,28,28),dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)
```

out：

<img src="/Users/dearr/Library/Application Support/typora-user-images/image-20240305155508959.png" alt="image-20240305155508959" style="zoom:50%;" />

<img src="/Users/dearr/Library/Application Support/typora-user-images/image-20240305161913339.png" alt="image-20240305161913339" style="zoom:50%;" />



针对解决的问题/使用的数据集？MNIST手写数字识别？



- 是谁（作者

Yann LeCun（AT&T 贝尔实验室）

- 在哪儿（期刊

<img src="/Users/dearr/Library/Application Support/typora-user-images/image-20240305161039390.png" alt="image-20240305161039390" style="zoom:50%;" />



- 啥时候（时间

1998年

<img src="/Users/dearr/Library/Application Support/typora-user-images/image-20240305160127335.png" alt="image-20240305160127335" style="zoom:50%;" />

- 优点（解决了过去的什么问题

**Part01**

LeNet是早期由Yann LeCun等人提出的卷积神经网络架构，被广泛用于手写数字识别等任务。LeNet具有以下优点：

1. 卷积结构：LeNet是最早引入卷积层和池化层的神经网络之一，利用卷积操作和参数共享的特性有效地减少了需要学习的参数数量，降低了模型复杂度。

2. 层级结构：LeNet通过多层卷积和池化层的叠加，实现了对输入数据特征的逐层提取和抽象，从而能够捕获不同层次的特征信息，提高了模型的表征能力。

3. 平移不变性：由于LeNet中采用了卷积操作和池化操作，使得网络对于输入数据的平移具有一定的不变性，即无论对象在图像中的位置如何变化，网络都可以识别相同的特征。

4. 效果良好：LeNet在早期的手写数字识别等任务上取得了较好的效果，证明了卷积神经网络在图像处理任务中的有效性和潜力。

5. 具有启发意义：LeNet的结构为后续更深、更复杂的卷积神经网络奠定了基础，为深度学习的发展提供了重要的启发和范例。

总的来说，LeNet作为卷积神经网络的先驱之一，具有卷积结构、层级结构、平移不变性、良好的效果和启发意义等优点，为深度学习和图像处理领域的发展做出了重要贡献。

**Part2**

CNN提出了三个创新点：局部感受野（Local Receptive Fields）、共享权重（Shared Weights）、时空下采样（Spatial or Temporal Subsampling）。

1. 局部感受野（Local Receptive Fields）：(是什么) 用来表示网络内部的不同位置的神经元对原图像的感受范围的大小，对应于CNN中的卷积核，可以抽取图像初级的特征，如边、转角等，这些特征会在后来的层中通过各种联合的方式来检测高级别特征。

   > **为什么提出局部感受野（解决了过去什么问题）**
   >
   > 局部感受野（local receptive field）是指神经网络中每个神经元接收的输入信息来自于输入数据的局部区域。通过使用局部感受野，神经元只会关注输入数据的一小部分区域，而不是整个输入数据，这样可以提供以下几方面的优势和解决过去的问题：
   >
   > 1. 参数共享：通过使用局部感受野，可以实现参数共享，即多个神经元共享相同的权重。这样可以减少模型的参数数量，降低过拟合的风险，并且加速模型的训练过程。
   > 2. 稀疏连接：局部感受野可以实现稀疏连接，即每个神经元仅与输入数据的局部区域连接。这样可以降低计算复杂度，并且使得神经网络更加高效。
   > 3. 特征提取：局部感受野有助于网络从局部区域提取特征，使得网络能够更好地捕获输入数据的空间层次结构和局部模式，提高网络对特定模式的感知能力。
   > 4. 增强平移不变性：局部感受野允许网络在不同位置学习到相同的特征，从而增强了网络对于平移不变性的学习能力，使得网络对于输入数据的位置变化更加鲁棒。
   >
   > 因此，通过使用局部感受野，神经网络可以更有效地提取特征、减少参数数量、增强特征的平移不变性，从而改善网络的性能并解决过去在图像处理和深度学习任务中遇到的问题。
   >
   > **卷积神经网络（CNN）提出局部感受野的概念**主要是为了模拟人类视觉系统的工作原理，并针对图像处理任务中的局部特征提取进行优化。以下是其中的原因和优势：
   >
   > 1. 局部特征提取：在图像处理任务中，不同位置的像素之间可能存在相关性和依赖关系。通过引入局部感受野，即卷积核的大小，网络可以专注于处理输入数据中的局部区域，并提取局部特征。这有助于网络更好地捕捉图像中的细节和局部模式。
   >
   > 2. 参数共享：局部感受野使得卷积操作在每个位置上都使用相同的权重进行特征提取，从而实现参数共享。这样可以大大减少需要学习的参数数量，降低了模型复杂度，并且提高了模型的泛化能力。
   >
   > 3. 平移不变性：通过使用局部感受野，卷积神经网络能够在整个输入图像上平移其学习到的特征。这意味着无论对象在图像中的位置如何变化，网络都可以捕捉到相同的特征。这种平移不变性使得网络能够更好地应对物体的位置变化和平移。
   >
   > 4. 多层次特征提取：卷积神经网络通常由多个卷积层和池化层组成，每一层都可以使用不同尺寸的局部感受野。这样可以逐渐扩大网络对于输入数据的感受野大小，从而实现多层次的特征提取，从低级到高级特征的逐步抽象。
   >
   > 综上所述，局部感受野的引入使得卷积神经网络能够更有效地处理图像数据，并且具备平移不变性、参数共享和多层次特征提取的优势。这使得卷积神经网络成为图像处理和计算机视觉任务中的重要工具。

2. (是什么)共享权重（Shared Weights）：在卷积过程中，每个卷积核所对应的窗口会以一定的步长在输入矩阵（图像）上不断滑动并进行卷积操作，最后，每个卷积核会生成一个对应的feature map（也就是卷积核的输出），一个feature map中的每一个单元都是由相同的权重（也就是对应的卷积核内的数值）计算得到的，这就是共享权重。

   > (为什么)卷积神经网络（CNN）提出共享权重的概念主要是为了解决以下问题和带来以下优势：
   >
   > 1. 参数数量：在传统的全连接神经网络中，每个神经元与上一层的所有神经元相连，导致参数数量巨大。而卷积神经网络通过共享权重，可以显著减少参数数量。共享权重意味着对于输入数据的不同位置，使用相同的权重进行特征提取，从而大大降低了需要学习的参数数量。
   >
   > 2. 特征提取：共享权重使得网络能够学习到对于输入数据的局部特征，而这些特征在整个输入数据中可能具有普遍性。通过共享权重，网络能够更好地捕获输入数据的局部模式，提高网络对特定模式的感知能力。
   >
   > 3. 平移不变性：共享权重使得网络在不同位置学习到相同的特征，从而增强了网络对于平移不变性的学习能力。这意味着无论对象出现在图像的哪个位置，网络都能够识别其特征。
   >
   > 4. 模型泛化：共享权重有助于减少过拟合的风险，因为参数共享使得模型更加简洁，更容易泛化到新的数据。
   >
   > 综上所述，共享权重的引入使得卷积神经网络在图像处理和其他领域取得了巨大的成功，大大减少了参数数量，提高了特征提取的效率，并增强了网络对于平移不变性的学习能力，从而成为处理视觉和空间数据的重要工具。
   >
   > 总之，共享权重通过减少参数数量、防止过拟合、提高数据利用率以及增强平移不变性等方面，改善了过去神经网络中存在的问题，并提升了模型的性能和效率。

3. 时空下采样（Spatial or Temporal Subsampling）：对应于CNN中的池化操作，也就是从卷积得到的feature map中提取出重要的部分，此操作可以降低模型对与图像平移和扭曲的敏感程度。

   > 卷积神经网络（CNN）引入了时空下采样的概念，主要是为了解决两个问题：局部平移不变性和降低特征图的维度。
   >
   > 1. 局部平移不变性：在图像识别等任务中，同一个对象可能出现在图像的不同位置，而我们希望模型能够识别出这个对象，而不受其具体位置的影响。通过时空下采样，即池化操作，可以使得特征图对于输入数据的小范围平移具有一定的鲁棒性，从而提高模型的平移不变性。
   >
   > 2. 降低特征图的维度：随着网络层数的增加，特征图的维度会逐渐增大，这样会导致参数数量的急剧增加，同时也增加了计算量。通过时空下采样，可以降低特征图的维度，减少计算量和内存消耗，同时提取出更为显著和重要的特征。
   >
   > 因此，时空下采样的引入有助于提高模型的鲁棒性、降低计算复杂度，并且能够在一定程度上防止过拟合。这些优点使得卷积神经网络在图像处理、语音识别等领域取得了巨大的成功。

- 缺点（未来的研究趋势

> - 目前的cnn一般是[CONV - RELU - POOL]，这里的[CONV - POOL - RELU]
> - 文章的激活函数是sigmoid，目前图像一般用tanh，relu，leakly relu较多，实践证明，一般比sigmoid好。
> - 目前，多分类最后一层一般用softmax，文中的与此不太相同。
> - ![image-20240305175724184](/Users/dearr/Library/Application Support/typora-user-images/image-20240305175724184.png)
> - 第二就是使用平均池化，现在一般主要使用最大池化，因为最大池化的效果更好。第三是在池化层之后引入了非线性，现在一般是在卷积层后通过激活函数获取非线性，在池化层后不再引入非线性；第四是LeNet-5最后一步是Gaussian Connections，目前已经被Softmax取代。
>
> 





# AlexNet



- 是什么（原理

  

  关系定义法

<img src="/Users/dearr/Library/Application Support/typora-user-images/image-20240305180743720.png" alt="image-20240305180743720" style="zoom:40%;" />

<img src="/Users/dearr/Library/Application Support/typora-user-images/image-20240305180842838.png" alt="image-20240305180842838" style="zoom:40%;" />







<img src="/Users/dearr/Library/Application Support/typora-user-images/image-20240305182235694.png" alt="image-20240305182235694" style="zoom:50%;" />



<img src="/Users/dearr/Library/Application Support/typora-user-images/image-20240305181046566.png" alt="image-20240305181046566" style="zoom:50%;" />



属性定义法

<img src="/Users/dearr/Library/Application Support/typora-user-images/image-20240305181456345.png" alt="image-20240305181456345" style="zoom:50%;" />

<img src="/Users/dearr/Library/Application Support/typora-user-images/image-20240305181508357.png" alt="image-20240305181508357" style="zoom:50%;" />

<img src="/Users/dearr/Library/Application Support/typora-user-images/image-20240305184813549.png" alt="image-20240305184813549" style="zoom:50%;" />

<img src="/Users/dearr/Library/Application Support/typora-user-images/image-20240305184925193.png" alt="image-20240305184925193" style="zoom:50%;" />

<img src="/Users/dearr/Library/Application Support/typora-user-images/image-20240305184941452.png" alt="image-20240305184941452" style="zoom:50%;" />

[经典卷积神经网络--AlexNet的详解-CSDN博客](https://blog.csdn.net/hgnuxc_1993/article/details/115840197)

什么是LRN？





- 为什么（bg 要解决的问题

  ImageNet 是一个超过1500万张带标签的高分辨率图像的数据集，这些图片归属于22,000多个类别。作为 PASCAL 视觉目标挑战赛的一部分，一年一度的 ImageNet 大型视觉识别挑战赛（ILSVRC）从2010年开始举办。ILSVRC 使用 ImageNet 的一个子集，分为1000种类别，每种类别中都有大约1000张图像。总之，大约有120万张训练图像，50,000张验证图像和150,000张测试图像。

  ILSVRC-2010是ILSVRC这个系列赛唯一一个给了测试集标签的版本，所以我们大部分实验都在这个版本上进行，在section 6上我们汇报了在该数据集上的结果。在 ImageNet 上，观察两个误差率：top-1 和 top-5 ，top-1就是直接看模型预测出来的和真实标签不同的百分比，其中 top-5 误差率是指测试图像上正确标签不属于被模型认为是最有可能的五个标签的百分比。

- 怎么做（代码

- 是谁（作者

AlexNet是由Alex Krizhevsky、Ilya Sutskever和Geoffrey Hinton（ImageNet图像分类竞赛中提出的一种经典的卷积神经网络）

- 在哪儿（期刊

- 啥时候（时间

  2012年

- 优点（解决了过去的什么问题、创新点）

- 缺点（未来的研究趋势













# summary

<img src="/Users/dearr/Library/Application Support/typora-user-images/image-20240305135415215.png" alt="image-20240305135415215" style="zoom:50%;" />





# VGG

是什么

一般画图都会省略  **输入图像→操作（卷积、池化、非线性激活）→输出**

![image-20240306095031238](/Users/dearr/Library/Application Support/typora-user-images/image-20240306095031238.png)

 <img src="/Users/dearr/Downloads/vgg.png" alt="vgg" style="zoom:80%;" />

<img src="/Users/dearr/Library/Application Support/typora-user-images/image-20240306131539204.png" alt="image-20240306131539204" style="zoom:50%;" />

需要注意的点

kernel size全是3  四周填充为1        $p_h = 2$ 

![image-20240306130841038](/Users/dearr/Library/Application Support/typora-user-images/image-20240306130841038.png)



VGG块

<img src="/Users/dearr/Library/Application Support/typora-user-images/image-20240306130717304.png" alt="image-20240306130717304" style="zoom:33%;" />

<img src="/Users/dearr/Library/Application Support/typora-user-images/image-20240306131706520.png" alt="image-20240306131706520" style="zoom:50%;" />

<img src="/Users/dearr/Library/Application Support/typora-user-images/image-20240306132115157.png" alt="image-20240306132115157" style="zoom:50%;" />



[快速理解VGG网络-CSDN博客](https://blog.csdn.net/weixin_44957722/article/details/119089221)

<img src="/Users/dearr/Library/Application Support/typora-user-images/image-20240306132148204.png" alt="image-20240306132148204" style="zoom:50%;" />

<img src="/Users/dearr/Library/Application Support/typora-user-images/image-20240306132256471.png" alt="image-20240306132256471" style="zoom:50%;" />

<img src="/Users/dearr/Library/Application Support/typora-user-images/image-20240306134501273.png" alt="image-20240306134501273" style="zoom:50%;" />

这是一个VGG块

李沐的书通过 操作定义VGG块

> VGG块的定义：
>
> 通过操作定义
>
> 通过输入输出定义



也许这个才是 输入和输出尺寸相同

<img src="/Users/dearr/Library/Application Support/typora-user-images/image-20240306134648951.png" alt="image-20240306134648951" style="zoom:50%;" />

<img src="/Users/dearr/Library/Application Support/typora-user-images/image-20240306132457471.png" alt="image-20240306132457471" style="zoom:50%;" />

**感受野相同该怎么解释？**

以块的角度来看  VGG

![image-20240306135226917](/Users/dearr/Library/Application Support/typora-user-images/image-20240306135226917.png)

（都一样 都一个东西）

不用管他这么说 那么说 都一样了

[经典网络架构学习-VGG_vgg网络结构-CSDN博客](https://blog.csdn.net/BXD1314/article/details/125781929?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1-125781929-blog-119089221.235^v43^pc_blog_bottom_relevance_base9&spm=1001.2101.3001.4242.2&utm_relevant_index=4)

- 是什么（原理
- 为什么（bg 要解决的问题
- 怎么做（代码
- 是谁（作者

这篇博文将介绍一下在ImageNet 2014 年斩获目标定位竞赛的第一名，图像分类竞赛的第二名的网络结构VGG。VGG 是 Visual Geometry Group 的缩写，是这个网络创建者的队名，作者来自牛津大学

- 在哪儿（期刊

VGG论文[《Very Deep Convolutional Networks for Large-Scale Image Recognition》](https://arxiv.org/pdf/1409.1556.pdf)作为一篇会议论文在2015年的ICLR大会上发表

Visual Geometry Group实验室链接：https://www.robots.ox.ac.uk/~vgg/

- 啥时候（时间
- 优点（解决了过去的什么问题
- 缺点（未来的研究趋势

# NIN

<img src="/Users/dearr/Library/Application Support/typora-user-images/image-20240306135735374.png" alt="image-20240306135735374" style="zoom:50%;" />
