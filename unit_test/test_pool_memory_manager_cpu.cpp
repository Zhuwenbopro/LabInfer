// 单元测试编译命令：g++ -std=c++17 -O2 -Wall -o run_tests test_pool_memory_manager_cpu.cpp ../src/PoolMemoryManager.cpp

#include "../include/PoolMemoryManager.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>
#include <stdexcept>

// 为了测试，我们需要一个模拟的内存块
std::vector<char> g_test_pool;

// 辅助函数，用于打印测试用例名称
void print_test_header(const std::string& test_name) {
    std::cout << "\n--- Running test: " << test_name << " ---" << std::endl;
}

// 测试1：基本分配和回收
void test_basic_allocation() {
    print_test_header("Basic Allocation and Deallocation");
    
    const size_t pool_size = 1024000000; // 100MB;
    g_test_pool.assign(pool_size, 0);
    PoolMemoryManager pool(g_test_pool.data(), pool_size);

    assert(pool.get_total_memory() == pool_size);
    assert(pool.get_free_memory() == pool_size);

    void* p1 = pool.allocate(100);
    assert(p1 != nullptr);
    assert(pool.get_free_memory() < pool_size - 100); // 考虑到对齐，可能大于100

    void* p2 = pool.allocate(200);
    assert(p2 != nullptr);

    pool.deallocate(p1);
    pool.deallocate(p2);

    // 完全释放后，空闲内存应恢复到总量
    assert(pool.get_free_memory() == pool_size);
    std::cout << "Test passed!" << std::endl;
}

// 测试2：边界条件和错误处理
void test_edge_cases() {
    print_test_header("Edge Cases and Error Handling");

    const size_t pool_size = 1024;
    g_test_pool.assign(pool_size, 0);
    PoolMemoryManager pool(g_test_pool.data(), pool_size);

    // 分配0字节
    void* p0 = pool.allocate(0);
    assert(p0 == nullptr);

    // 分配整个池
    void* p_full = pool.allocate(pool_size);
    assert(p_full != nullptr);
    assert(pool.get_free_memory() == 0);
    pool.deallocate(p_full);
    assert(pool.get_free_memory() == pool_size);

    // 尝试分配超大内存
    bool caught_exception = false;
    try {
        pool.allocate(pool_size + 1);
    } catch (const std::bad_alloc& e) {
        caught_exception = true;
    }
    assert(caught_exception);

    // 释放一个无效指针 (这里我们简单地忽略cerr的输出)
    int invalid_ptr_val = 42;
    void* invalid_ptr = &invalid_ptr_val;
    pool.deallocate(invalid_ptr); // 应该在stderr打印错误，但程序不应崩溃

    // 重复释放
    void* p_double_free = pool.allocate(50);
    assert(p_double_free != nullptr);
    pool.deallocate(p_double_free);
    pool.deallocate(p_double_free); // 第二次释放应该被安全地处理

    std::cout << "Test passed!" << std::endl;
}

// 测试3：核心功能 - 空闲块合并
void test_coalescing() {
    print_test_header("Coalescing (Block Merging)");

    const size_t pool_size = 1024;
    const size_t alignment = 256;
    g_test_pool.assign(pool_size, 0);
    PoolMemoryManager pool(g_test_pool.data(), pool_size, alignment);
    
    // 我们将分配3个块 A, B, C，然后释放 A 和 C，再释放 B，看 B 是否能与 A, C 合并
    // 对齐大小为 256
    void* pA = pool.allocate(100); // 实际分配 256
    void* pB = pool.allocate(200); // 实际分配 256
    void* pC = pool.allocate(300); // 实际分配 512 (因为请求300 > 256)
    
    assert(pool.get_free_memory() == 0); // 256+256+512 = 1024

    // 场景1：释放中间块 B，前后都是已分配块，不应发生合并
    pool.deallocate(pB);
    // 空闲内存应为 256。此时池中有三个块：[Allocated A], [Free 256], [Allocated C]
    assert(pool.get_free_memory() == 256); 
    
    // 场景2：释放后面的块 C，它应该与前面刚释放的 B 合并
    // 先把 B 重新分配回来
    pB = pool.allocate(200);
    assert(pool.get_free_memory() == 0);
    // 释放顺序 C -> B
    pool.deallocate(pC); // Free C (512)
    assert(pool.get_free_memory() == 512);
    pool.deallocate(pB); // Free B (256), should merge with C
    // 此时空闲块应该是 [Allocated A], [Free 768]
    assert(pool.get_free_memory() == 512 + 256);
    
    // 场景3：释放前面的块 A，它应该与后面的 (B+C) 空闲块合并，恢复整个池
    pool.deallocate(pA);
    assert(pool.get_free_memory() == pool_size); // 256 + 768 = 1024

    std::cout << "Test passed!" << std::endl;
}

// 测试4：碎片化场景
void test_fragmentation_and_reuse() {
    print_test_header("Fragmentation and Reuse");
    
    const size_t pool_size = 1024;
    g_test_pool.assign(pool_size, 0);
    PoolMemoryManager pool(g_test_pool.data(), pool_size, 1); // alignment=1 方便计算
    
    // 制造碎片: alloc(A), alloc(B), alloc(C), alloc(D)
    // 然后 free(B), free(D)
    // 内存布局: [A-used] [hole1] [C-used] [hole2]
    void* pA = pool.allocate(100);
    void* pB = pool.allocate(200);
    void* pC = pool.allocate(300);
    void* pD = pool.allocate(400); // 100+200+300+400 = 1000, 剩余24
    
    pool.deallocate(pB); // [hole1] size 200
    pool.deallocate(pD); // [hole2] size 400

    assert(pool.get_free_memory() == 200 + 400 + 24);

    // 尝试分配一个需要大连续空间的块，应该失败
    bool caught = false;
    try {
        pool.allocate(500);
    } catch(const std::bad_alloc&) {
        caught = true;
    }
    assert(caught);

    // 尝试分配一个可以放入 hole2 的块
    void* p_reuse1 = pool.allocate(350);
    assert(p_reuse1 != nullptr);
    assert(pool.get_free_memory() == 200 + (400 - 350) + 24); // 200 + 50 + 24 = 274

    // 尝试分配一个可以放入 hole1 的块
    void* p_reuse2 = pool.allocate(150);
    assert(p_reuse2 != nullptr);
    assert(pool.get_free_memory() == (200 - 150) + 50 + 24); // 50 + 50 + 24 = 124

    // 释放所有，检查是否能恢复
    pool.deallocate(pA);
    pool.deallocate(pC);
    pool.deallocate(p_reuse1);
    pool.deallocate(p_reuse2);
    assert(pool.get_free_memory() == pool_size);
    
    std::cout << "Test passed!" << std::endl;
}

int main() {
    std::cout << "========== Starting PoolMemoryManager Unit Tests ==========" << std::endl;

    try {
        test_basic_allocation();
        test_edge_cases();
        test_coalescing();
        test_fragmentation_and_reuse();
    } catch (const std::exception& e) {
        std::cerr << "A test failed with an unhandled exception: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n========== All tests completed successfully! ==========" << std::endl;
    
    return 0;
}