# LABInfer
<p align="center">
  <img src="./assets/logo.png" width="300" alt="LABINFER">
</p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;目标是创建异构、分布式大语言模型推理框架。尽量使用最直白的实现方式实现各个模块，并提供设计实现思路，方便各位实验室的同学在此基础上修改代码，融入自己的 idea。

**TODO List**

- [ ] llama 3 中用到的函数（cuda 实现）。
- [ ] cuda allocator。
- [ ] device 为上层提供接口服务。


- [ ] 分词器 tokenizer
- [ ] cuda支持16位精度浮点运算

**Finished List**


## 设计
* **设计目标**：异构的、可调度的、分布式的推理服务程序。

<p align="center">
  <img src="./assets/arch.png" width="500" alt="架构">
</p>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;整体架构如上图所示，项目分为三个阶段完成。首先完成设备抽象层的实现，到此层有抽象的接口去使用底层不同硬件；然后实现管理层代码（中间两部分），实现后可以在单机多卡上跑通大语言模型；最后一个阶段是实现分布式的推理，完成后可以一键提供大语言模型的推理服务。

### Device 层
#### Allocator
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Allocator 类目前就负责两个责任：1）分配设备内存，2）回收设备内存。更高级、复制的机制我们在后面再慢慢往里加。
```
Allocator
    void allocate(void* ptr, std::size_t size) = 0;
    void deallocate(void* ptr) = 0;
```
#### Function


#### 变量（variable）与模型（model）的关系
变量类 Variable 有两个子类：Tensor、Parameter，分别代表隐藏层之间传递的参数和模型自身的权重。变量应该仅为模型使用，整个模型的内存也都由变量占据。变量的内存应该由模型借助 allocator 去分配。这样在进行模型内部改造时会比较灵活。*为了节省内存，可以让一块GPU内的所有隐藏层结果放到同一块显存中。*
#### 模型类的设计
由于本框架只面向推理，模型的结构一早就构建好了。使用者不应该关系模型如何构建，设计理想是根据一张模型的图纸（*model.map*），在服务启动时就会自动地构建好模型，加载好参数，准备开始服务。用户在使用模型时，只需要提供模型参数和模型图纸即可。

#### 变量类的设计
对于变量内部的device，总共就那么几种，不需要每个变量都开辟新的空间，可以用某个设计模式解决这个问题。

#### 如何实现异构
异构，即同一套代码可以使用不同的加速器，在本框架下主要体现在两个部分 —— allocator、function，分别提供数据的存储和运算。



## 学习笔记
#### 如何读取权重？（权重文件的格式）
对于 huggingface 的 xxx.safetensor，详细的介绍请看[这里](https://zhuanlan.zhihu.com/p/686570419)
#### 矩阵在 cuda 里是怎么表示的？（cuda里都是一位的数组，是怎么分割成矩阵，并进行运算的）

