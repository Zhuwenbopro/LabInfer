#include "CPU/CPUMemoryManager.h"
#include <cstdlib>

void* CPUMemoryManager::allocate(size_t size){
    return malloc(size);
}

void CPUMemoryManager::deallocate(void* ptr){
    free(ptr);
    ptr = nullptr;
}