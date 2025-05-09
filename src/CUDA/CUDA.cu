#include "device/CUDA/CUDA.h"
#include "CUDAUtils.h"
#include "allocator/CUDAAllocator.h"
// #include "CUDAFunctionFactory.h"

CUDA::CUDA(int id) {
    this->id_ = id;
    
    this->allocator_ = std::unique_ptr<CUDAAllocator>();

    CUDA_CHECK(cudaSetDevice(this->id_));
    // this->func = create_cuda_function(dtype);
}

CUDA::~CUDA() {
    CUDA_CHECK(cudaDeviceReset());
}

void CUDA::move_in(std::shared_ptr<void> ptr_dev, std::shared_ptr<void> ptr_cpu, size_t bytes) {
    CUDA_CHECK(cudaMemcpy(ptr_dev.get(), ptr_cpu.get(), bytes, cudaMemcpyHostToDevice));
}

void CUDA::move_out(std::shared_ptr<void> ptr_dev, std::shared_ptr<void> ptr_cpu, size_t bytes) {
    CUDA_CHECK(cudaMemcpy(ptr_cpu.get(), ptr_dev.get(), bytes, cudaMemcpyDeviceToHost));
}

void CUDA::copy(std::shared_ptr<void> dst, std::shared_ptr<void> src, size_t bytes) {
    CUDA_CHECK(cudaMemcpy(dst.get(), src.get(), bytes, cudaMemcpyDeviceToDevice));
}