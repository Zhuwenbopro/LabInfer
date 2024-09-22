// Allocator.h
#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include <cstddef>

class Allocator {
public:
    virtual ~Allocator() {}
    
    // 分配内存
    virtual void* allocate(std::size_t size) = 0;
    
    // 回收内存
    virtual void deallocate(void* ptr) = 0;
};

#endif // ALLOCATOR_H
