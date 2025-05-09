#pragma once

#include <memory>

class Allocator {
public:
    virtual ~Allocator() = default;

    virtual std::shared_ptr<void> allocate(size_t bytes) = 0;
};