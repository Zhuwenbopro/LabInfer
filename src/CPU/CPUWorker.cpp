#include "CPU/CPUWorker.h"

void CPUWorker::handle_infer(Command cmd) {
    // std::cout << "[Worker " << get_id() << " TID: " << get_thread_id_str() << "] Handling part of INFER (ReqID: " << cmd.request_id << ") input: '" << cmd.input_data << "'" << std::endl;
    // if (!is_initialized_)
    // {
    //     Result res;
    //     res.request_id = cmd.request_id;
    //     res.success = false;
    //     res.error_message = "Worker " + std::to_string(get_id()) + " not initialized for INFER.";
    //     // This worker failed its part. If it's the last one, it sets the master promise.
    //     if (cmd.remaining_workers->fetch_sub(1, std::memory_order_acq_rel) == 1)
    //     {
    //         std::cout << "[Worker " << get_id() << " TID: " << get_thread_id_str() << "] (ReqID: " << cmd.request_id << ") Last worker, setting master promise (due to its own init error)." << std::endl;
    //         cmd.master_promise->set_value(res);
    //     }
    //     else
    //     {
    //         std::cout << "[Worker " << get_id() << " TID: " << get_thread_id_str() << "] (ReqID: " << cmd.request_id << ") Not last worker, error reported for its part." << std::endl;
    //     }
    //     return;
    // }

    // // Simulate this worker's part of TP inference
    // std::this_thread::sleep_for(std::chrono::milliseconds(200 + (cmd.request_id % 3) * 50 + get_id() * 20));
    // std::string partial_output = cmd.input_data;
    // // Each worker could do something different if we were truly simulating TP
    // // For demo, let's just say worker 0 is responsible for the "final" string reversal
    // if (get_id() == 0)
    // {
    //     std::reverse(partial_output.begin(), partial_output.end());
    //     partial_output = "[W0_reversed] " + partial_output;
    // }
    // else
    // {
    //     partial_output = "[W" + std::to_string(get_id()) + "_processed] " + partial_output;
    // }

    // std::cout << "[Worker " << get_id() << " TID: " << get_thread_id_str() << "] Part of INFER complete (ReqID: " << cmd.request_id << "). Partial output: '" << partial_output << "'" << std::endl;

    // // Decrement counter. If this worker is the last one, it sets the master promise.
    // if (cmd.remaining_workers->fetch_sub(1, std::memory_order_acq_rel) == 1)
    // {
    //     std::cout << "[Worker " << get_id() << " TID: " << get_thread_id_str() << "] (ReqID: " << cmd.request_id << ") This is the LAST worker for this INFER task. Setting master promise." << std::endl;
    //     Result final_res;
    //     final_res.request_id = cmd.request_id;
    //     final_res.success = true;
    //     // For simplicity, the last worker to finish provides its output
    //     // In a real TP, results would be gathered and aggregated by the Engine or a designated worker.
    //     // Here, we could decide worker 0's result is the "main" one, or some aggregation.
    //     // Let's just use this worker's (the last one) partial output as the final output for demo.
    //     final_res.output_data = "Aggregated (simulated by last worker " + std::to_string(get_id()) + "): " + partial_output;
    //     cmd.master_promise->set_value(final_res);
    // }
    // else
    // {
    //     std::cout << "[Worker " << get_id() << " TID: " << get_thread_id_str() << "] (ReqID: " << cmd.request_id << ") Not the last worker for this INFER task. (" << cmd.remaining_workers->load() << " remaining)" << std::endl;
    // }
}

void CPUWorker::handle_init(Command cmd) {
    // std::cout << "[Worker " << get_id() << " TID: " << get_thread_id_str() << "] Handling INIT command..." << std::endl;

    // 模拟初始化过程
    // std::this_thread::sleep_for(std::chrono::milliseconds(100 + get_id() * 50));
    is_initialized_ = true;
    // model_name_ = "SimulatedModel_v1.0_Worker" + std::to_string(get_id());
    // std::cout << "[Worker " << get_id() << " TID: " << get_thread_id_str() << "] Initialized with model: " << model_name_ << std::endl;

    
    try {
        Result res;
        res.success = true;
        res.output_data = "Worker " + std::to_string(get_id()) + " initialized successfully.";
        cmd.individual_promise.set_value(res);
    } catch (const std::future_error &e) {
        // 处理设置值时的错误
        std::cerr << "[Worker " << get_id() << " TID: " << get_thread_id_str()
                  << "] Future error setting value for INIT promise: " << e.what() << std::endl;
    }
}