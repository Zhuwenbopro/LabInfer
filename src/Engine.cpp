#include "Engine.h"
#include <sstream>
#include <iomanip>
#include "CUDA/CUDAWorker.h"

Engine::Engine(int num_workers) : num_workers_(num_workers), request_id_counter_(0)
{
    if (num_workers <= 0)
        throw std::invalid_argument("Number of workers must be positive.");
    std::cout << "[Engine] Creating " << num_workers_ << " workers." << std::endl;
    for (int i = 0; i < num_workers_; ++i)
    {
        // TODO: 根据硬件环境选择合适的Worker类型
        workers_.emplace_back(std::make_unique<CUDAWorker>(i, this));
    }
}

Engine::~Engine()
{
    std::cout << "[Engine] Destructor called. Shutting down workers..." << std::endl;
    shutdown_workers();
    workers_.clear();
    std::cout << "[Engine] All workers shut down." << std::endl;
}

std::string get_thread_id_str()
{
    std::stringstream ss;
    ss << std::hex << std::setw(8) << std::setfill('0') << std::this_thread::get_id();
    return ss.str();
}

void Engine::initialize_workers()
{
    std::cout << "[Engine] Initializing all workers..." << std::endl;
    std::vector<std::future<Result>> init_futures;

    for (auto &worker_ptr : workers_)
    {
        worker_ptr->start();
        std::promise<Result> init_promise;
        init_futures.push_back(init_promise.get_future());
        // INIT is still an individual command per worker
        Command init_cmd(CommandType::INIT, std::move(init_promise));
        worker_ptr->push_command(std::move(init_cmd));
    }

    for (size_t i = 0; i < init_futures.size(); ++i)
    {
        try
        {
            Result res = init_futures[i].get();
            if (res.success)
            {
                std::cout << "[Engine] Worker " << i << " initialization result: " << res.output_data << std::endl;
            }
            else
            {
                std::cerr << "[Engine] Worker " << i << " initialization failed: " << res.error_message << std::endl;
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "[Engine] Exception for worker " << i << " init: " << e.what() << std::endl;
        }
    }
    std::cout << "[Engine] All workers initialized (or init attempted)." << std::endl;
}

void Engine::shutdown_workers()
{
    std::cout << "[Engine] Sending SHUTDOWN to all workers..." << std::endl;
    for (auto &worker_ptr : workers_)
    {
        worker_ptr->stop(); // stop now sends SHUTDOWN and joins
    }
    std::cout << "[Engine] All workers stopped." << std::endl;
}

std::future<Result> Engine::submit_inference_request(const std::string &input_text) {
    uint64_t req_id = request_id_counter_++;

        // Create one master promise for this entire multi-worker request
        auto master_promise_ptr = std::make_shared<std::promise<Result>>();
        std::future<Result> result_future = master_promise_ptr->get_future();

        // Counter for how many workers need to complete this task
        // For TP, this would be the number of workers in the TP group. Here, all workers.
        auto remaining_workers_ptr = std::make_shared<std::atomic<int>>(num_workers_);

        std::cout << "[Engine TID: " << get_thread_id_str() << "] Submitting INFER (ReqID: " << req_id << ") to ALL " << num_workers_ << " workers." << std::endl;

        for (auto &worker_ptr : workers_)
        {
            Command infer_sub_cmd(CommandType::INFER, req_id, input_text,
                                  master_promise_ptr, remaining_workers_ptr);
            worker_ptr->push_command(std::move(infer_sub_cmd));
        }
        return result_future;
}