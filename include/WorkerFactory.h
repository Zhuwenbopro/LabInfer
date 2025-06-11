#pragma once

#include "common.h"
#include "CUDA/CUDAWorker.h"

class Engine;
std::unique_ptr<Worker> create_worker(HardwareType hardware_type, int worker_id, Engine* engine) {
    switch (hardware_type)
    {
        case HardwareType::CUDA:
            return std::make_unique<CUDAWorker>(worker_id, engine);
        // case HardwareType::CPU:
        //     return std::make_unique<CPUWorker>(worker_id, engine);
        // case HardwareType::METAL:
        //     return std::make_unique<MetalWorker>(worker_id, engine);
        default:
            throw std::runtime_error("Unsupported hardware type in WorkerFactory.");
    }
}