#include "CUDA.h"
#include "CUDAUtils.h"
#include "CUDAAllocator.h"
#include "CUDAFunction.h"

// 构造函数：初始化 CUDA 对象，并设置对应的 CUDA 设备
CUDA::CUDA(int id)
{
    device_id = id;
    CUDA_CHECK(cudaSetDevice(device_id));
    device_name = "CUDA";
    allocator = new CUDAAllocator();
    F = new CUDAFunction();
}

// 析构函数：释放 CUDA 对象占用的资源
CUDA::~CUDA()
{
    delete allocator;
    delete F;
}

// 从主机（CPU）内存拷贝数据到设备（GPU）内存
void CUDA::move_in(std::shared_ptr<void> ptr_dev, std::shared_ptr<void> ptr_cpu, size_t bytes)
{
    CUDA_CHECK(cudaMemcpy(ptr_dev.get(), ptr_cpu.get(), bytes, cudaMemcpyHostToDevice));
}

// 从设备（GPU）内存拷贝数据到主机（CPU）内存
void CUDA::move_out(std::shared_ptr<void> ptr_dev, std::shared_ptr<void> ptr_cpu, size_t bytes)
{
    CUDA_CHECK(cudaMemcpy(ptr_cpu.get(), ptr_dev.get(), bytes, cudaMemcpyDeviceToHost));
}

// 在设备（GPU）上分配内存，返回分配内存的指针-
std::shared_ptr<void> CUDA::allocate(size_t bytes)
{
    return allocator->allocate(bytes);
}

// 在设备（GPU）之间进行内存拷贝（设备到设备）
void CUDA::copy(std::shared_ptr<void> dst, std::shared_ptr<void> src, size_t bytes)
{
    CUDA_CHECK(cudaMemcpy(dst.get(), src.get(), bytes, cudaMemcpyDeviceToDevice));
}
