#include "CPU/CPUWorker.h"

Result CPUWorker::handle_infer(std::shared_ptr<Batch> batch) 
{
    return Result();
}

Result CPUWorker::handle_init() 
{
    is_initialized_ = true;


    return Result();
}