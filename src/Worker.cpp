#include "../include/Worker.h"

// 初始化的时候把模型架子搭好，等着load param
// TODO: 读取配置文件，加载模型
Worker::Worker(int id, Engine *engine) : id_(id), engine_(engine), running_(false)
{

}

Worker::~Worker()
{
    stop();
    std::cout << "[Worker " << get_id() << "] Destroyed." << std::endl;
}

void Worker::start()
{
    if (running_)
        return;
    running_ = true;
    thread_ = std::thread(&Worker::process_loop, this);
}

void Worker::stop()
{
    if (!running_)
        return;

    running_ = false;
    Command shutdown_cmd(CommandType::SHUTDOWN);
    command_queue_.push(std::move(shutdown_cmd));
    if (thread_.joinable())
    {
        thread_.join();
    }
    std::cout << "[Worker " << get_id() << "] Thread stopped." << std::endl;
}

void Worker::push_command(Command cmd){
    command_queue_.push(std::move(cmd));
}


void Worker::process_loop()
{
    while (running_)
    {
        Command cmd = command_queue_.pop(); // 从队列获取命令

        // 1. 根据命令类型，调用相应的处理函数来执行任务并获取结果
        Result task_result;
        switch (cmd.type)
        {
        case CommandType::INIT:
            task_result = this->handle_init();
            break;
        case CommandType::INFER:
            task_result = this->handle_infer(cmd.batch);
            break;
        case CommandType::SHUTDOWN:
            std::cout << "[Worker " << get_id() << "] Received SHUTDOWN. Exiting loop." << std::endl;
            running_ = false;
            // SHUTDOWN 通常没有 promise，直接跳到下一次循环（这将导致循环结束）
            continue; 
        default:
            task_result.success = false;
            task_result.error_message = "Unknown command type!";
            std::cerr << "[Worker " << get_id() << "] " << task_result.error_message << std::endl;
        }

        // 如果任务失败，设置全局失败标志
        if (!task_result.success && cmd.any_worker_failed) {
            cmd.any_worker_failed->store(true, std::memory_order_relaxed);
        }

        // 2. 统一处理回调逻辑
        if (cmd.completion_promise && cmd.remaining_tasks)
        {
            // 原子地减少计数器，并检查自己是否是最后一个
            if (cmd.remaining_tasks->fetch_sub(1, std::memory_order_acq_rel) == 1)
            {
                // 我是最后一个，我负责最终的报告
                Result final_res;
                final_res.request_id = cmd.request_id;
                
                // 检查是否有任何 worker 失败了
                bool any_failed = cmd.any_worker_failed ? cmd.any_worker_failed->load() : !task_result.success;

                if (any_failed) {
                    final_res.success = false;
                    // TODO: 可以设计更复杂的机制来聚合所有错误信息
                    final_res.error_message = "One or more workers failed the task.";
                } else {
                    final_res.success = true;
                    // 如果是 INFER，这里应该做结果聚合
                    // 为了简化，我们只使用最后一个 worker 的结果作为最终结果
                    if (cmd.type == CommandType::INFER) {
                        final_res.output_data = "Aggregated result (simulated): " + task_result.output_data;
                    } else {
                        final_res.output_data = "Group task completed successfully.";
                    }
                }
                
                std::cout << "[Worker " << get_id() << "] Last worker, setting master promise." << std::endl;
                cmd.completion_promise->set_value(final_res);
            }
        }
    }
    std::cout << "[Worker " << get_id() << "] Processing loop finished." << std::endl;
}


std::string Worker::get_thread_id_str()
{
    std::stringstream ss;
    ss << std::hex << std::setw(8) << std::setfill('0') << std::this_thread::get_id();
    return ss.str();
}
