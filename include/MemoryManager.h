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
};