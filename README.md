# LABInfer
<p align="center">
  <img src="./assets/logo.png" width="300" alt="LABINFER">
</p>

[roadmap](https://github.com/users/Zhuwenbopro/projects/1)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;目标是创建异构、分布式大语言模型推理框架。尽量使用最直白的实现方式实现各个模块，并提供设计实现思路，方便各位实验室的同学在此基础上修改代码，融入自己的 idea。

```
# 安装 OpenBLAS
sudo apt-get install libopenblas-dev

sudo apt-get install libopencv-dev
```

# 组织架构
![结构图](./assets/arch.png)

# TODO
- [] 进行原型验证（5.7 - 5.11）
  - [] 整体 Engine 启动，Worker初始化Layer、Device、Communicator完成。（5.7）
  - [] 数据流动完成，Communicator 完成通信。（5.8）
- [] 技术调研（5.10 - 5.18）
- [] 技术原型验证（5.12 - 5.23）
- [] 细节实现（5.24 - 5.31）
- [] 创新功能（To be continue）

## Engine
* 创建模型的运行结构 workers
* 初始化 worker （装上device、layer初始化，装参数）
