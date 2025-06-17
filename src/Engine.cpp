#include "Engine.h"
#include "WorkerFactory.h"


Engine::Engine(int num_workers, DeviceType device_type) : num_workers_(num_workers), device_type_(device_type), request_id_counter_(0)
{
    if (num_workers <= 0)
        throw std::invalid_argument("Number of workers must be positive.");
    std::cout << "[Engine] Creating " << num_workers_ << " workers." << std::endl;
    for (int i = 0; i < num_workers_; ++i)
    {
        // TODO: 根据硬件环境选择合适的Worker类型
        workers_.emplace_back(create_worker(CUDA, i, this));
    }
}

Engine::~Engine()
{
    std::cout << "[Engine] Destructor called. Shutting down workers..." << std::endl;
    shutdown_workers();
    workers_.clear();
    std::cout << "[Engine] All workers shut down." << std::endl;
}

std::future<Result> Engine::submit_group_command(const Command& command_template)
{
    auto promise_ptr = std::make_shared<std::promise<Result>>();
    auto counter_ptr = std::make_shared<std::atomic<int>>(num_workers_);
    auto failed_flag_ptr = std::make_shared<std::atomic<bool>>(false);
    
    // 从 promise 获取 future，这是将要返回给调用者的
    std::future<Result> result_future = promise_ptr->get_future();
    
    uint64_t req_id = request_id_counter_++;

    for (auto &worker_ptr : workers_)
    {
        Command cmd_copy = command_template;

        cmd_copy.request_id = req_id;
        cmd_copy.completion_promise = promise_ptr;
        cmd_copy.remaining_tasks = counter_ptr;
        cmd_copy.any_worker_failed = failed_flag_ptr;

        worker_ptr->push_command(std::move(cmd_copy));
    }

    return result_future;
}

void Engine::initialize_workers()
{
    std::cout << "[Engine] Starting all worker threads..." << std::endl;
    for (auto &worker_ptr : workers_) {
        worker_ptr->start();
    }

    std::cout << "[Engine] Submitting INIT command to all workers..." << std::endl;
    Command init_template(CommandType::INIT);
    
    try
    {
        Result res = submit_group_command(init_template).get();
        if (res.success)
        {
            std::cout << "[Engine] All workers initialized successfully." << std::endl;
        } else
        {
            std::cerr << "[Engine] Overall worker initialization failed: " << res.error_message << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Engine] Exception during initialization process: " << e.what() << std::endl;
    }
}

void Engine::shutdown_workers()
{
    std::cout << "[Engine] Sending SHUTDOWN to all workers..." << std::endl;
    for (auto &worker_ptr : workers_)
    {
        worker_ptr->stop();
    }
    std::cout << "[Engine] All workers stopped." << std::endl;
}

std::future<Result> Engine::submit_inference_request(const std::string &input_text)
{
    Command infer_template(CommandType::INFER);

    return submit_group_command(infer_template);
}