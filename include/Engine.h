#pragma once

#include "common.h"
#include "Worker.h"
#include <future>

class Engine
{
public:
    Engine(int num_workers) : num_workers_(num_workers), request_id_counter_(0)
    {
        if (num_workers <= 0)
            throw std::invalid_argument("Number of workers must be positive.");
        std::cout << "[Engine] Creating " << num_workers_ << " workers." << std::endl;
        for (int i = 0; i < num_workers_; ++i)
        {
            workers_.emplace_back(std::make_unique<Worker>(i, this));
        }
    }

    ~Engine()
    {
        std::cout << "[Engine] Shutting down..." << std::endl;
        shutdown_workers();
        workers_.clear();
        std::cout << "[Engine] Shutdown complete." << std::endl;
    }

    void initialize_workers();

    void shutdown_workers();

    // MODIFIED to dispatch to all workers for INFER
    std::future<Result> submit_inference_request(const std::string &input_text);

private:
    int num_workers_;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::atomic<uint64_t> request_id_counter_;
};