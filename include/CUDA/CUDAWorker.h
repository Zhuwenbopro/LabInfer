#pragma once

#include "Worker.h"
#include "CUDA/CUDAWorker.h"


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
};