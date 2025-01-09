#ifndef CUDA_H
#define CUDA_H

#include "Device.h"

// CUDA 子类
class CUDA : public Device {
public:
    CUDA();
    ~CUDA();
    void move_in(void* ptr_dev, void* ptr_cpu, size_t bytes) override;
    void move_out(void* ptr_dev, void* ptr_cpu, size_t bytes) override;
    void* allocate(size_t bytes) override;
    void deallocate(void* ptr) override;
    void copy(void* dst, void* src, size_t bytes) override;
};

#endif // CUDA_H