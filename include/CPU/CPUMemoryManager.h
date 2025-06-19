#pragma once

#include "MemoryManager.h"

class CPUMemoryManager : public MemoryManager {
public:
    CPUMemoryManager() { }

    void* allocate(size_t size) override;

    void deallocate(void* ptr) override;

    void move_in(void* ptr_dev, void* ptr_cpu, size_t bytes) override;

    void move_out(void* ptr_dev, void* ptr_cpu, size_t bytes) override;
};