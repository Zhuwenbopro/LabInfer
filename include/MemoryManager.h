#pragma once
#include <cstddef>


// 未来优化方向：实现内存池（申请大块内存自行管理），减少内存分配和释放的开销
class MemoryManager {
public:
    virtual ~MemoryManager() = default;
    
    // 分配内存
    virtual void* allocate(size_t size) = 0;
    
    // 回收内存
    virtual void deallocate(void* ptr) = 0;

    virtual void move_in(void* ptr_dev, void* ptr_cpu, size_t bytes) = 0;

    virtual void move_out(void* ptr_dev, void* ptr_cpu, size_t bytes) = 0;

    // 获取可用内存大小
    // virtual size_t get_free_memory() const = 0;

    // 获取总内存大小
    // virtual size_t get_total_memory() const = 0;
};