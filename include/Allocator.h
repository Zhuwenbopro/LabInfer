// Allocator.h
#ifndef ALLOCATOR_H
#define ALLOCATOR_H
#include <memory>

class Allocator {
public:
    virtual ~Allocator() = default;
    
    // 分配内存
    virtual std::shared_ptr<void> allocate(size_t size) = 0;
};

#endif // ALLOCATOR_H
