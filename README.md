# LABInfer
<p align="center">
  <img src="./assets/logo.png" width="300" alt="LABINFER">
</p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;目标是创建异构、分布式大语言模型推理框架。尽量使用最直白的实现方式实现各个模块，并提供设计实现思路，方便各位实验室的同学在此基础上修改代码，融入自己的 idea。

**TODO List**
#### 第一阶段 v0.1 (已完成)
分词器 tokenizer : encode、decode 暂时先用 [huggingface](https://github.com/huggingface/tokenizers) 的吧，之后再说
- [x] 各种 transformer 层 layers
- [x] state load -- [safetensor]([https://github.com/syoyo/safetensors-cpp)
- [x] 手工搭一个[模型测试推理](https://huggingface.co/meta-llama/Llama-3.2-1B/tree/main?library=transformers)
- [x] 支持 kv cache
- [x] 支持 batch

```
/test/layer_test 内的 测试程序 main.cc 需要下面的数据，放到文件下
链接：https://pan.baidu.com/s/1NaCiVebIFhUJZ60hkm851g?pwd=b72y 
提取码：b72y 
```

#### 第二阶段
- [ ] swap
- [ ] 每层采样计算时间，用于后续评估
- [ ] pipeline
- [ ] pagedAttention
- [ ] chunk
- [ ] 根据采样时间划分模型
