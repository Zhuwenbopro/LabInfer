#pragma once

#include "Worker.h"
#include "CUDA/CUDAMemoryManager.h"


class CUDAWorker : public Worker
{
public:
    CUDAWorker(int id, Engine *engine);
    ~CUDAWorker();

private:
    Result handle_init() override;
    Result handle_infer(std::shared_ptr<Batch> batch) override;

    CUDAMemoryManager memory_manager_;
};