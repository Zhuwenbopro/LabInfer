#include "../include/Worker.h"

// 初始化的时候把模型架子搭好，等着load param
// TODO: 读取配置文件，加载模型
Worker::Worker(int id, Engine *engine) : id_(id), engine_(engine), running_(false)
{
    std::cout << "[Worker " << get_id() << "] Created." << std::endl;
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
    std::cout << "[Worker " << get_id() << "] Thread started (ID: " << get_thread_id_str() << ")." << std::endl;
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
        Command cmd = command_queue_.pop();

        switch (cmd.type)
        {
        case CommandType::INIT:
            this->handle_init(std::move(cmd));
            break;
        case CommandType::INFER:
            this->handle_infer(std::move(cmd));
            break;
        case CommandType::SHUTDOWN:
            std::cout << "[Worker " << get_id() << " TID: " << get_thread_id_str() << "] Received SHUTDOWN. Exiting loop." << std::endl;
            running_ = false;
            // Fulfill promise if it was provided with shutdown
            if (cmd.individual_promise.get_future().valid())
            {
                Result res;
                res.success = true;
                cmd.individual_promise.set_value(res);
            }
            else if (cmd.master_promise && cmd.master_promise->get_future().valid())
            {
                // This case should not happen for a typical SHUTDOWN.
                // But if it did, we'd need to handle the counter.
                Result res;
                res.success = true;
                // This is complex for shutdown, as other workers might still be running.
                // For simplicity, SHUTDOWN is treated as an individual command here.
                // cmd.master_promise->set_value(res);
            }
            return;
        default:
            std::cerr << "[Worker " << get_id() << " TID: " << get_thread_id_str() << "] Unknown command type!" << std::endl;
            // Fulfill any promise to avoid deadlocks
            if (cmd.individual_promise.get_future().valid())
            {
                Result res;
                res.success = false;
                res.error_message = "Unknown command for individual promise";
                cmd.individual_promise.set_value(res);
            }
            else if (cmd.master_promise && cmd.master_promise->get_future().valid() && cmd.remaining_workers)
            {
                if (cmd.remaining_workers->fetch_sub(1, std::memory_order_acq_rel) == 1)
                {
                    Result res;
                    res.success = false;
                    res.error_message = "Unknown command for master promise";
                    cmd.master_promise->set_value(res);
                }
            }
        }
    }
    std::cout << "[Worker " << get_id() << " TID: " << get_thread_id_str() << "] Processing loop finished." << std::endl;
}


std::string Worker::get_thread_id_str()
{
    std::stringstream ss;
    ss << std::hex << std::setw(8) << std::setfill('0') << std::this_thread::get_id();
    return ss.str();
}
