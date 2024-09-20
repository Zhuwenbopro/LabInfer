# LABInfer
<p align="center">
  <img src="./assets/image.png" width="300" alt="LABINFER">
</p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;目标是创建异构、分布式大语言模型推理框架。尽量使用最直白的实现方式实现各个模块，方便各位实验室的同学在此基础上修改代码，融入自己的 idea。

**TODO List**

- [ ] 模型层类代码实现
- [ ] 把 llama 3 用到的cuda函数做成动态库
- [ ] 分词器 tokenizer
- [ ] 内存分配器 allocator
- [ ] cuda支持16位精度浮点运算

**Finished List**


## 设计
#### 变量（variable）与模型（model）的关系
变量类 Variable 有两个子类：Tensor、Parameter，分别代表隐藏层之间传递的参数和模型自身的权重。变量应该仅为模型使用，整个模型的内存也都由变量占据。变量的内存应该由模型借助 allocator 去分配。这样在进行模型内部改造时会比较灵活。*为了节省内存，可以让一块GPU内的所有隐藏层结果放到同一块显存中。*
#### 模型类的设计
由于本框架只面向推理，模型的结构一早就构建好了。使用者不应该关系模型如何构建，设计理想是根据一张模型的图纸（*model.map*），在服务启动时就会自动地构建好模型，加载好参数，准备开始服务。用户在使用模型时，只需要提供模型参数和模型图纸即可。

#### 变量类的设计
对于变量内部的device，总共就那么几种，不需要每个变量都开辟新的空间，可以用某个设计模式解决这个问题。
## 学习笔记
#### 如何读取权重？（权重文件的格式）
对于 huggingface 的 xxx.safetensor，详细的介绍请看[这里](https://zhuanlan.zhihu.com/p/686570419)
#### 矩阵在 cuda 里是怎么表示的？（cuda里都是一位的数组，是怎么分割成矩阵，并进行运算的）

