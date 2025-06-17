// nvcc -std=c++17 -o run_gpu_tests_nvcc test_pool_memory_manager_cuda.cu ../src/PoolMemoryManager.cpp

#include "../include/PoolMemoryManager.h" // 仅依赖 PoolMemoryManager
#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>
#include <stdexcept>
#include <cuda_runtime.h> // 需要直接使用 CUDA API

// 辅助宏，检查 CUDA 调用是否成功
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__,         \
                    __LINE__, cudaGetErrorString(err));                   \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

// 辅助函数，用于打印测试用例名称
void print_test_header(const std::string& test_name) {
    std::cout << "\n--- Running test: " << test_name << " ---" << std::endl;
}

// 获取当前可用的 GPU 显存
size_t get_gpu_free_mem() {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    return free_mem;
}

// 测试夹具类（Test Fixture）来管理 GPU 内存的生命周期
class PoolOnGpuTestFixture {
public:
    PoolOnGpuTestFixture(size_t pool_size) : size_(pool_size) {
        if (get_gpu_free_mem() < size_) {
            throw std::runtime_error("Not enough free GPU memory to run test.");
        }
        // 1. 手动申请 GPU 显存
        CUDA_CHECK(cudaMalloc(&gpu_pool_ptr_, size_));
        // 2. 用这块显存初始化 PoolMemoryManager
        pool_manager_ = new PoolMemoryManager(gpu_pool_ptr_, size_);
        std::cout << "Test fixture created a " << size_ / (1024*1024) << "MB pool on GPU." << std::endl;
    }

    ~PoolOnGpuTestFixture() {
        // 3. 析构 PoolMemoryManager
        delete pool_manager_;
        // 4. 手动释放 GPU 显存
        if (gpu_pool_ptr_) {
            CUDA_CHECK(cudaFree(gpu_pool_ptr_));
        }
        std::cout << "Test fixture destroyed the GPU pool." << std::endl;
    }

    // 提供对 PoolMemoryManager 的访问
    PoolMemoryManager& get_pool() {
        return *pool_manager_;
    }

private:
    void* gpu_pool_ptr_ = nullptr;
    size_t size_;
    PoolMemoryManager* pool_manager_ = nullptr;
};

// 测试1：基本分配和验证
void test_basic_gpu_interaction() {
    print_test_header("Basic Interaction on GPU");
    const size_t pool_size = 256 * 1024 * 1024; // 256MB
    
    PoolOnGpuTestFixture fixture(pool_size);
    PoolMemoryManager& pool = fixture.get_pool();
    
    assert(pool.get_total_memory() == pool_size);
    assert(pool.get_free_memory() == pool_size);

    // 从池中分配 10MB
    const size_t alloc_size = 10 * 1024 * 1024;
    void* d_ptr = pool.allocate(alloc_size);
    assert(d_ptr != nullptr);
    
    // 验证分配的 GPU 指针是有效的
    // 创建主机数据
    std::vector<int> host_data(1024, 42); // 4KB of data
    // 拷贝到 GPU
    CUDA_CHECK(cudaMemcpy(d_ptr, host_data.data(), host_data.size() * sizeof(int), cudaMemcpyHostToDevice));
    // 从 GPU 拷贝回来进行验证
    std::vector<int> host_data_verify(1024);
    CUDA_CHECK(cudaMemcpy(host_data_verify.data(), d_ptr, host_data_verify.size() * sizeof(int), cudaMemcpyDeviceToHost));
    
    assert(host_data == host_data_verify);

    pool.deallocate(d_ptr);
    assert(pool.get_free_memory() == pool_size);
    
    std::cout << "Test passed!" << std::endl;
}

// 测试2：GPU上的合并逻辑
void test_gpu_coalescing_standalone() {
    print_test_header("Coalescing Logic on GPU addresses");
    const size_t pool_size = 128 * 1024 * 1024; // 128MB
    
    PoolOnGpuTestFixture fixture(pool_size);
    PoolMemoryManager& pool = fixture.get_pool();

    void* pA = pool.allocate(20 * 1024 * 1024);
    void* pB = pool.allocate(40 * 1024 * 1024);
    void* pC = pool.allocate(60 * 1024 * 1024);
    
    // 此时内存池几乎用完
    size_t remaining_mem = pool.get_free_memory();

    // 释放中间的 pB
    pool.deallocate(pB);
    
    // 尝试分配一个比 pB 小的块，应该成功
    void* pB_new = pool.allocate(30 * 1024 * 1024);
    assert(pB_new != nullptr);

    // 全部释放，顺序：A -> C -> B_new
    pool.deallocate(pA);
    pool.deallocate(pC);
    pool.deallocate(pB_new);

    // 验证所有内存都已合并并归还
    assert(pool.get_free_memory() == pool_size);

    std::cout << "Test passed!" << std::endl;
}


int main() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA-capable device found. Skipping GPU tests." << std::endl;
        return 0;
    }
    
    std::cout << "========== Starting Standalone PoolMemoryManager on GPU Tests ==========" << std::endl;

    try {
        test_basic_gpu_interaction();
        test_gpu_coalescing_standalone();
    } catch (const std::exception& e) {
        std::cerr << "A test failed with an unhandled exception: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n========== All tests completed successfully! ==========" << std::endl;
    
    return 0;
}