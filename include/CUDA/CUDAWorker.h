#pragma once

#include "Worker.h"
#include "CUDA/CUDAMemoryManager.h"


class CUDAWorker : public Worker
{
public:
    CUDAWorker(int id, Engine *engine) : Worker(id, engine)
    {
        std::cout << "[CUDAWorker " << get_id() << "] Created." << std::endl;
    }

    ~CUDAWorker()
    {
        stop();
        std::cout << "[CUDAWorker " << get_id() << "] Destroyed." << std::endl;
    }
private:
    void handle_init(Command cmd) override;

    CUDAMemoryManager memory_manager_;
};