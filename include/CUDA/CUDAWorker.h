#pragma once

#include "Worker.h"
#include "CUDA/CUDAMemoryManager.h"


class CUDAWorker : public Worker
{
public:
    CUDAWorker(int id, Engine *engine);
    ~CUDAWorker();

private:
    void handle_init(Command cmd) override;
    void handle_infer(Command cmd) override;

    CUDAMemoryManager memory_manager_;
};