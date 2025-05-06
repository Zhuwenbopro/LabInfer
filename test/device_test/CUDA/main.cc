#include <iostream>
#include <thread>
#include <memory>
#include <cstring>      // for memset
#include "CUDA.h"       // 包含 CUDA 类和其它相关声明

// 每个线程执行的函数：使用特定设备 id 进行一些简单的 CUDA 操作测试
void cuda_thread_function(int device_id) {
    // 创建对应设备的 CUDA 对象（构造函数中会调用 cudaSetDevice）
    CUDA cuda(device_id);

    // 输出线程的 CUDA 设备信息
    std::cout << "Thread for device " << device_id << " started." << std::endl;

    // 示例操作：分配一定大小的设备内存，并用主机数据拷贝进来
    const size_t elementCount = 1024;
    const size_t bytes = elementCount * sizeof(float);

    // 在设备上分配内存
    std::shared_ptr<void> devPtr = cuda.allocate(bytes);
    if (!devPtr) {
        std::cerr << "Failed to allocate device memory on device " << device_id << std::endl;
        return;
    }

    // 在主机上分配内存并初始化数据
    float* hostData = new float[elementCount];
    for (size_t i = 0; i < elementCount; ++i) {
        hostData[i] = static_cast<float>(i);
    }
    // 使用 shared_ptr<float> 管理该数组，再转换为 shared_ptr<void>
    std::shared_ptr<float> tempHostPtr(hostData, std::default_delete<float[]>());
    std::shared_ptr<void> hostPtr = std::static_pointer_cast<void>(tempHostPtr);

    // 通过 move_in() 将数据从主机拷贝到设备
    cuda.move_in(devPtr, hostPtr, bytes);
    std::cout << "Device " << device_id << ": Data moved from host to device." << std::endl;

    // 清空主机数据（仅作为测试），然后将数据从设备拷贝回主机检查正确性
    memset(hostData, 0, bytes);
    cuda.move_out(devPtr, hostPtr, bytes);
    std::cout << "Device " << device_id << ": Data moved back from device to host." << std::endl;

    // 验证结果（简单输出部分数据）
    float* result = static_cast<float*>(hostPtr.get());
    std::cout << "Device " << device_id << " first five elements: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    // 当 cuda 对象离开作用域时，其析构函数将自动释放创建的资源
}

int main() {
    // 创建两个线程，每个线程分别控制一个 CUDA 设备
    // 注意：确保你的系统至少拥有两个 CUDA 设备，否则会出现 cudaSetDevice 错误
    std::thread thread0(cuda_thread_function, 0);
    std::thread thread1(cuda_thread_function, 1);

    // 等待所有线程完成
    thread0.join();
    thread1.join();

    std::cout << "Both CUDA threads have completed their operations." << std::endl;
    return 0;
}
