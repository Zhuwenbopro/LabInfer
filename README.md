# LABInfer
<p align="center">
  <img src="./assets/logo.png" width="300" alt="LABINFER">
</p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;目标是创建异构、分布式大语言模型推理框架。尽量使用最直白的实现方式实现各个模块，并提供设计实现思路，方便各位实验室的同学在此基础上修改代码，融入自己的 idea。

```
# 安装 OpenBLAS
sudo apt-get install libopenblas-dev

sudo apt-get install libopencv-dev
```


## 已完成
- llama 3.1流程推理
- 采样：top-k、top-p、temperature
- 计算优化：kv cache
- load [safetensor]([https://github.com/syoyo/safetensors-cpp)


```
/test/layer_test 内的 测试程序 main.cc 需要下面的数据，放到文件下
链接：https://pan.baidu.com/s/1NaCiVebIFhUJZ60hkm851g?pwd=b72y 
提取码：b72y 
```

## TODO List
- [ ] 性能评估：运行时间、吞吐量、占用空间
- [ ] 多设备支持
- [ ] tensor parallel
- [ ] pagedAttention
- [ ] chunk
- [ ] benchmark 测试设备性能
- [ ] 自动化模型划分
