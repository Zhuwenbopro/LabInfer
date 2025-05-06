#ifndef CUDA_H
#define CUDA_H

#include "Device.h"

class CUDA : public Device {
private:
    
public:
    CUDA(int id = 0);
    ~CUDA();
    void move_in(std::shared_ptr<void> ptr_dev, std::shared_ptr<void> ptr_cpu, size_t bytes) override;
    void move_out(std::shared_ptr<void> ptr_dev, std::shared_ptr<void> ptr_cpu, size_t bytes) override;
    std::shared_ptr<void> allocate(size_t bytes) override;
    void copy(std::shared_ptr<void> dst, std::shared_ptr<void> src, size_t bytes) override;
};


#endif