#include "../include/Worker.h"

void Worker::start(){
    if (running_)
        return;
    running_ = true;
    thread_ = std::thread(&Worker::process_loop, this);
    std::cout << "[Worker " << get_id() << "] Thread started (ID: " << get_thread_id_str() << ")." << std::endl;
}

void Worker::stop(){
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

void Worker::process_loop(){
    while (running_)
    {
        Command cmd = command_queue_.pop();

        switch (cmd.type)
        {
        case CommandType::INIT:
            handle_init(std::move(cmd));
            break;
        case CommandType::INFER:
            handle_infer(std::move(cmd));
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

void Worker::handle_init(Command cmd){
    std::cout << "[Worker " << get_id() << " TID: " << get_thread_id_str() << "] Handling INIT command..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(100 + get_id() * 50));
    is_initialized_ = true;
    model_name_ = "SimulatedModel_v1.0_Worker" + std::to_string(get_id());
    std::cout << "[Worker " << get_id() << " TID: " << get_thread_id_str() << "] Initialized with model: " << model_name_ << std::endl;

    Result res;
    res.success = true;
    res.output_data = "Worker " + std::to_string(get_id()) + " initialized successfully.";
    try
    {
        cmd.individual_promise.set_value(res);
    }
    catch (const std::future_error &e)
    {
        // This might happen if set_value is called more than once,
        // or if the future was detached, etc.
        // For INIT, this should ideally not happen if logic is correct.
        std::cerr << "[Worker " << get_id() << " TID: " << get_thread_id_str()
                  << "] Future error setting value for INIT promise: " << e.what() << std::endl;
    }
}

void Worker::handle_infer(Command cmd){
    std::cout << "[Worker " << get_id() << " TID: " << get_thread_id_str() << "] Handling part of INFER (ReqID: " << cmd.request_id << ") input: '" << cmd.input_data << "'" << std::endl;
    if (!is_initialized_)
    {
        Result res;
        res.request_id = cmd.request_id;
        res.success = false;
        res.error_message = "Worker " + std::to_string(get_id()) + " not initialized for INFER.";
        // This worker failed its part. If it's the last one, it sets the master promise.
        if (cmd.remaining_workers->fetch_sub(1, std::memory_order_acq_rel) == 1)
        {
            std::cout << "[Worker " << get_id() << " TID: " << get_thread_id_str() << "] (ReqID: " << cmd.request_id << ") Last worker, setting master promise (due to its own init error)." << std::endl;
            cmd.master_promise->set_value(res);
        }
        else
        {
            std::cout << "[Worker " << get_id() << " TID: " << get_thread_id_str() << "] (ReqID: " << cmd.request_id << ") Not last worker, error reported for its part." << std::endl;
        }
        return;
    }

    // Simulate this worker's part of TP inference
    std::this_thread::sleep_for(std::chrono::milliseconds(200 + (cmd.request_id % 3) * 50 + get_id() * 20));
    std::string partial_output = cmd.input_data;
    // Each worker could do something different if we were truly simulating TP
    // For demo, let's just say worker 0 is responsible for the "final" string reversal
    if (get_id() == 0)
    {
        std::reverse(partial_output.begin(), partial_output.end());
        partial_output = "[W0_reversed] " + partial_output;
    }
    else
    {
        partial_output = "[W" + std::to_string(get_id()) + "_processed] " + partial_output;
    }

    std::cout << "[Worker " << get_id() << " TID: " << get_thread_id_str() << "] Part of INFER complete (ReqID: " << cmd.request_id << "). Partial output: '" << partial_output << "'" << std::endl;

    // Decrement counter. If this worker is the last one, it sets the master promise.
    if (cmd.remaining_workers->fetch_sub(1, std::memory_order_acq_rel) == 1)
    {
        std::cout << "[Worker " << get_id() << " TID: " << get_thread_id_str() << "] (ReqID: " << cmd.request_id << ") This is the LAST worker for this INFER task. Setting master promise." << std::endl;
        Result final_res;
        final_res.request_id = cmd.request_id;
        final_res.success = true;
        // For simplicity, the last worker to finish provides its output
        // In a real TP, results would be gathered and aggregated by the Engine or a designated worker.
        // Here, we could decide worker 0's result is the "main" one, or some aggregation.
        // Let's just use this worker's (the last one) partial output as the final output for demo.
        final_res.output_data = "Aggregated (simulated by last worker " + std::to_string(get_id()) + "): " + partial_output;
        cmd.master_promise->set_value(final_res);
    }
    else
    {
        std::cout << "[Worker " << get_id() << " TID: " << get_thread_id_str() << "] (ReqID: " << cmd.request_id << ") Not the last worker for this INFER task. (" << cmd.remaining_workers->load() << " remaining)" << std::endl;
    }
}
