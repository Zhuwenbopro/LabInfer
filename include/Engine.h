#pragma once

#include "common.h"
#include "Worker.h"
#include <future>

class Engine
{
public:
    Engine(int num_workers, DeviceType device_type);

    ~Engine();

    void initialize_workers();

    void shutdown_workers();

    std::future<Result> submit_inference_request(Batch& batch);

private:
    int num_workers_;
    DeviceType device_type_;

    std::vector<std::unique_ptr<Worker>> workers_;
    std::atomic<uint64_t> request_id_counter_;

    std::future<Result> submit_group_command(const Command& command_template);
};