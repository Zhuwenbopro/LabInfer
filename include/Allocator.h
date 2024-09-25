// Allocator.h
/**
    根据编译选项编译，因为不见得你调用的库那个机器上都有
 */
#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include <cstddef>

class Allocator {
public:
    virtual ~Allocator() = default;
    
    // 分配内存
    virtual void* allocate(size_t size) = 0;
    
    // 回收内存
    virtual void deallocate(void* ptr) = 0;
};

#endif // ALLOCATOR_H
