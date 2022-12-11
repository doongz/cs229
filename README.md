# CS229: Machine Learning

## Introduction

- 所属大学：Stanford
- 授课老师：Andrew Ng
- 先修要求：高数，概率论，需要较深厚的数学功底
- 编程语言：Python
- 课程难度：🌟🌟🌟🌟
- 预计学时：A month
- 学年：Fall 2022-23

吴恩达讲授，一门研究生课程，所以更偏重数学理论；

不满足于调包而想深入理解算法本质，或者有志于从事机器学习理论研究的同学可以学习这门课程。

课程网站上提供了所有的课程 notes，写得非常专业且理论，需要一定的数学功底。

**Course Description**

This course provides a broad introduction to machine learning and statistical pattern recognition. Topics include: 

- supervised learning (generative/discriminative learning, parametric/non-parametric learning, neural networks, support vector machines); 
- unsupervised learning (clustering, dimensionality reduction, kernel methods); 
- learning theory (bias/variance tradeoffs, practical advice); 
- reinforcement learning and adaptive control. 

The course will also discuss recent applications of machine learning, such as to robotic control, data mining, autonomous navigation, bioinformatics, speech recognition, and text and web data processing.

## Resources

- 课程网站：http://cs229.stanford.edu
- 课程视频：2018年 https://www.bilibili.com/video/BV1JE411w7Ub
- 课程教材：主要看 [main_notes.pdf](./main_notes.pdf)
- 课程作业：参见课程网站
- 实验参考：[PKUFlyingPig/CS229 - GitHub](https://github.com/PKUFlyingPig/CS229)

## Notes

### Part I Supervised learning

- **Linear regression**

线性：两个变量之间的关系**是**一次函数关系的—图象**是直线**，叫做线性；

回归：人们在测量事物的时候因为客观条件所限，求得的都是测量值，而不是事物真实的值，为了能够得到真实值，无限次的进行测量，最后通过这些测量数据计算**回归到真实值**，这就是回归的由来。

- **Classification and logistic regression**

逻辑回归，是一种广义的线性回归分析模型，属于机器学习中的监督学习。其推导过程与计算方式类似于回归的过程，但实际上主要是用来解决二分类问题（也可以解决多分类问题）。通过给定的n组数据（训练集）来训练模型，并在训练结束后对给定的一组或多组数据（测试集）进行分类。其中每一组数据都是由p 个指标构成。

- **Generalized linear models**

广义线性模型，旨在解决普通线性回归模型无法处理因变量离散，并发展能够解决非正态因变量的回归建模任务的建模方法。

在广义线性模型的框架下，因变量**不再要求连续、正态**，当然自变量更加没有特殊的要求。能够对**正态分布、二项分布、泊松分布、Gamma分布**等随机因变量进行建模。

- **Generative learning algorithms**

从训练集中我们先总结出各个类别的特征分布都是什么样的，也就是得到p(x|y)，然后再把要预测的样本特征拿来和每个类别的模型作比较，看它更像是哪种类别的样本。这种算法被称为**生成学习算法（generative learning algorithms）**。

- **Kernel methods**

支持向量机（SVM）是机器学习中一个常见的算法，通过最大间隔的思想去求解一个优化问题，得到一个分类超平面。对于非线性问题，则是通过引入核函数，对特征进行映射（通常映射后的维度会更高），在映射之后的特征空间中，样本点就变得线性可分了。

- **Support vector machines**

通俗来讲，所谓支持向量机是一种分类器，对于做出标记的两组向量，给出一个最优分割超曲面把这两组向量分割到两边，使得两组向量中离此超平面最近的向量（即所谓支持向量）到此超平面的距离都尽可能远。

### Part II Deep learning

- **Supervised learning with non-linear models**

- **Neural networks**

- **Backpropagation**

- **Vectorization over training examples**

训练样本集向量化。在进行深度学习领域之前，小规模数据使用for循环可能就足够用了，可是对现代的深度学习网络和当前规模的数据集来说，算法有更高的算力开销。如果用了for循环，代码运行就会很慢。

这就需要使用向量化了。向量化和for循环不同，能够利用矩阵线性代数的优势，还能利用一些高度优化的数值计算的线性代数包（比如BLAS），因此能使神经网络的计算运行更快。

### Part III Generalization and regularization

正则化（regularization）是所有用来降低算法泛化误差（generalization error）的方法的总称。正则化的手段多种多样，是以提升 bias 为代价降低 variance。现实中效果最好的深度学习模型，往往是【复杂的模型（大且深）】+【有效的正则化】。

- **Generalization**

- **Regularization and model selection**

### Part IV Unsupervised learning

- **Clustering and the k-means algorithm**

聚类(clustering)属于非监督学习(unsupervised learning) 无类别标记( class label)

Kmeans算法是最常用的聚类算法，主要思想是:在给定K值和K个初始类簇中心点的情况下，把每个点(亦即数据记录)分到离其最近的类簇中心点所代表的类簇中，所有点分配完毕之后，根据一个类簇内的所有点重新计算该类簇的中心点(取平均值)，然后再迭代的进行分配点和更新类簇中心点的步骤，直至类簇中心点的变化很小，或者达到指定的迭代次数。

- **EM algorithms**

最大期望算法，出现在很多涉及概率模型的计算生物学的应用中

概率模型，例如隐马尔可夫模型或者贝叶斯网络会被用于建模生物学数据。它们因为高效以及高容错的参数学习因此很流行被广泛应用。但是，给概率模型训练的数据往往不是完整的，缺失数据经常会发生。譬如，在医学诊断，通常病人的患病历史只会有有限的医学测试。换句话说，在基因的聚类表达中，数据不完整是因为在这个概率模型中有意忽略了基因到聚类的分配产生。然而最大期望算法支持在概率模型**用不完整的数据进行参数估计**。

- **Principal components analysis**

主成分分析（PCA），旨在利用**降维的思想**，把多指标转化为少数几个综合指标，同时保持数据集的对方差贡献最大的特征。在统计学中，PCA 是一种简化数据集的技术。

它是一个**线性变换**。这个变换把数据变换到一个新的坐标系统中，使得任何数据投影的第一大方差在第一个坐标(称为第一主成分)上，第二大方差在第二个坐标(第二主成分)上，依次类推。

- **Independent components analysis**

独立成分分析（ICA），将多元信号分解为不同的非高斯信号，也是要找到一组新的基向量来表征样本数据。它侧重于独立来源，与寻求最大化数据点方差的主成分分析不同。

噪声对记录的信号有很大的影响，不能从测量中去除。很难记录干净的测量值，因为源信号总是受到噪声和其他源提供的其他独立信号的污染。 因此，测量结果可能被描述为几个独立来源的汇编。盲源分离是分离这些混合信号（BSS）的过程。盲这个词表示即使关于它们的信息很少，源信号也可能被分离。独立成分分析 (ICA) 方法尝试从单个项目中识别或提取声音，尽管周围环境中存在其他噪音。

ICA 已用于在各种应用中提取源信号，包括医疗信号、生物测试和音频信号。当 ICA 可以删除或维护单个源时，它也称为降维算法。在此活动期间可以过滤或删除一些信号，这也称为过滤操作。

- **Self-supervised learning**

自监督学习，是无监督学习里面的一种，主要是希望能够学习到一种**通用的特征表达**用于**下游任务 (Downstream Tasks)**。 其主要的方式就是通过自己监督自己。

在预训练阶段我们使用**无标签的数据集 (unlabeled data)**，因为有标签的数据集**很贵**，打标签得要多少人工劳力去标注，那成本是相当高的。相反，无标签的数据集网上随便到处爬，它**便宜**。在训练模型参数的时候，我们不追求把这个参数用带标签数据从**初始化的一张白纸**给一步训练到位，原因就是数据集太贵。

于是**Self-Supervised Learning**就想先把参数从**一张白纸**训练到**初步成型**，再从**初步成型**训练到**完全成型**。注意这是2个阶段。这个**训练到初步成型的东西**，我们把它叫做**Visual Representation**。预训练模型的时候，就是模型参数从**一张白纸**到**初步成型**的这个过程，还是用无标签数据集。等我把模型参数训练个八九不离十，这时候再根据你**下游任务 (Downstream Tasks)**的不同去用带标签的数据集把参数训练到**完全成型**，那这时用的数据集量就不用太多了，因为参数经过了第1阶段就已经训练得差不多了。

- **foundation models**

AI专家**将大模型统一命名为Foundation Models**，论文肯定了Foundation Models对智能体基本认知能力的推动作用，同时也指出大模型**呈现出「涌现」与「同质化」的特性**。

所谓「涌现」代表一个系统的行为是隐性推动的，而不是显式构建的；「同质化」是指基础模型的能力是智能的中心与核心，大模型的任何一点改进会迅速覆盖整个社区，但其缺陷也会被所有下游模型所继承。

### Part V Reinforcement Learning and Control

- **Reinforcement learning**

强化学习，研究如何通过一系列的顺序决策来达成一个特定目标。广义地说，任何目标导向的问题都可以形式化为一个强化学习问题。

强化学习是一类算法, 是让计算机实现从一开始什么都不懂, 通过不断地尝试, 从错误中学习, 最后找到规律, 学会了达到目的的方法. 这就是一个完整的强化学习过程。

强化学习是一个大家族, 包含很多种算法, 一些比较有名的算法, 比如有通过行为的价值来选取特定行为的方法, 包括使用表格学习的 q learning, sarsa, 使用神经网络学习的 deep q network, 还有直接输出行为的 policy gradients, 又或者了解所处的环境, 想象出一个虚拟的环境并从虚拟的环境中学习 等等.

- **LQR, DDP and LQG**

LQR：线性二次调节，通过该模型我们可以求得精确的解。该模型常用于机器人控制，很多问题经常将问题简化成该模型，对于很多问题，即便其动态非线性，也可以化简为 LQR，例如倒立摆问题。

DDP：微分动态规划，之前所说的方法适用于优化目标为保持在某个状态 s⋆ 附近，如倒立摆、无人驾驶（保持在路中间）等。而某些情况下，目标往往更加复杂。微分动态规划这种方法，其适用于系统需要遵循某种轨迹（比如火箭）。该方法将轨迹离散化为离散的时间步，并创造中间目标来使用之前的方法。

LQG：线性二次高斯分布，目前为止，我们假设状态都是可以得到的，而在现实世界中，实际的观测值可能并不是真实的状态值（类似 HMM）。我们将使用「部分可观测 MDP」（POMDP）来解决这类问题。POMDP 是一种包含额外观察层的 MDP，在该框架下，一种通用的策略是先基于观测值 o1,…,ot 得到一个「置信状态」，然后 POMDP 的策略将置信状态映射为动作。本节将对 LQR 进行拓展来求解 POMDP。

- **Policy Gradient** (REINFORCE)

Policy Gradient是RL中一个大家族，是加上一个神经网络来输出预测的动作，不像Value-based方法（Q-Learning，Sarsa），但他也要接收环境信息（observation），不同的是他要输出不是action的value，而是具体的那一个action，这样Policy Gradient就跳过了value这个阶段。Policy Gradient对比起以值为基础的方法，Policy Gradient直接输出动作的最大好处就是，他能在一个连续区间内挑选动作，而基于值的，比如Q-Learning，它如果在无穷多得动作种计算价值，从而选择行为，这可吃不消。

Policy Gradient最大的一个优势是：输出的这个action可以是一个连续的值，之前我们说到的value-based方法输出的都是不连续的值，然后再选择值最大的action。而Policy Gradient可以在一个连续分布上选取action。

有了神经网络当然方便，但是，我们怎么进行神经网络的误差反向传递呢？Policy Gradient的误差又是什么呢？答案是没有误差。但是他的确是在进行某一种的反向传递。这种反向传递的目的是让这次被选中的行为更有可能在下次发生。但是我们要怎么确定这个行为是不是应当被增加被选的概率呢？这时候，reward奖惩正可以在这个时候排上用场。

## Lecture

- Lecture 1: [Introduction](./Lecture/Lecture-1-Introduction)
- Lecture 2: Supervised learning setup. LMS. `Sections 1.1, 1.2 of main notes`
- Lecture 3: Weighted Least Squares. Logistic regression. Newton's Method `Sections 1.3, 1.4, 2,1, 2.3 of main notes`
- Lecture 4: Dataset split; Exponential family. Generalized Linear Models. `Section 2.2 and Chapter 3 of main notes`
- Lecture 5: Gaussian discriminant analysis. Naive Bayes. `Section 4.1, 4.2 of main notes`
- Lecture 6: Naive Bayes, Laplace Smoothing.
- Lecture 7: Kernels; SVM `Chapter 5`
- Lecture 8: Neural Networks 1 `Sections 7.1, 7.2`
- Lecture 9: Neural Networks 2 (backprop) `Section 7.3`
- Lecture 10: [Bias-variance tradeoff, regularization](./Lecture/Lecture-10-Bias-variance-tradeoff®ularization) `Sections 8.1, 9.1, 9.3`
- Lecture 11: [Decision trees](./Lecture/Lecture-11-Decision-trees)
- Lecture 12: Boosting
- Lecture 13: [K-Means. GMM. Expectation Maximization.](./Lecture/Lecture-13-K-Means)
- Lecture 14: EM, PCA
- Lecture 15: [ML Advice](./Lecture/Lecture-15-ML-Advice)
- Lecture 16: [Other learning settings. Large language models & foundation models](./Lecture/Lecture-16-Other-learning-settings)
- Lecture 17: Basic concepts in RL, value iteration, policy iteration.
- Lecture 18: Model-based RL, value function approximator
- Lecture 19: fairness, algorithmic bias, explainability, privacy
- Lecture 20: fairness, algorithmic bias, explainability, privacy

---

- TA Lecture 1: [Linear Algebra Review](./TALecture/TALecture-1-Linear-Algebra)
- TA Lecture 2: [Probability Review](./TALecture/TALecture-2-Probability)
- TA Lecture 3: [Python/Numpy](./TALecture/TALecture-3-Python-Numpy)
- TA Lecture 4: [Evaluation Metrics](./TALecture/TALecture-4-Evaluation-Metrics)
- TA Lecture 5: [Midterm Review](./TALecture/TALecture-5-Midterm)
- TA Lecture 6: Deep Learning (Convnets)
- TA Lecture 7: [GANs](./TALecture/TALecture-7-GANs)
