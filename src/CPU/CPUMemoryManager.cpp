#include "CPU/CPUMemoryManager.h"
#include <cstdlib>
#include <stdexcept>

void* CPUMemoryManager::allocate(size_t size){
    return malloc(size);
}

void CPUMemoryManager::deallocate(void* ptr){
    free(ptr);
    ptr = nullptr;
}

void CPUMemoryManager::move_in(void* ptr_dev, void* ptr_cpu, size_t bytes) {
    throw std::logic_error("不应该把 CPU 里面的数据移入到 CPU 里!");
}

void CPUMemoryManager::move_out(void* ptr_dev, void* ptr_cpu, size_t bytes) {
    throw std::logic_error("不应该把 CPU 里面的数据移出到 CPU 里!");
}