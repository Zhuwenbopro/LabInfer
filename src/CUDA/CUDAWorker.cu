#include "CUDA/CUDAWorker.h"
#include "CUDA/CUDAUtils.h"
// 删掉
#include "Batch.h"

cublasHandle_t handle;

CUDAWorker::CUDAWorker(int id, Engine *engine) : Worker(id, engine)
{
    std::cout << "[CUDAWorker " << get_id() << "] Created." << std::endl;
    LinearFuncPtr linear_func = OpRegistry::Instance().Linear().Get(CUDA, FLOAT32);
    model_ = std::make_unique<Linear>(linear_func);
}

CUDAWorker::~CUDAWorker()
{
    stop();
    std::cout << "[CUDAWorker " << get_id() << "] Destroyed." << std::endl;
}

void CUDAWorker::handle_init(Command cmd)
{
    // 1. 获取设备上下文
    // std::cout << "[Worker " << get_id() << " TID: " << get_thread_id_str() << "] Handling INIT command..." << std::endl;
    CUDA_CHECK(cudaSetDevice(get_id()));
    is_initialized_ = true;
    // void *d_array = memory_manager_.allocate(1000000 * sizeof(float));
    
    // 2. 装载参数
    // std::this_thread::sleep_for(std::chrono::seconds(500));

    // 3. 传回信号
    try
    {
        Result res;
        res.success = true;
        res.output_data = "Worker " + std::to_string(get_id()) + " initialized successfully.";
        cmd.individual_promise.set_value(res);
    }
    catch (const std::future_error &e)
    {
        std::cerr << "[Worker " << get_id() << " TID: " << get_thread_id_str()
                  << "] Future error setting value for INIT promise: " << e.what() << std::endl;
    }
}

void CUDAWorker::handle_infer(Command cmd)
{
    // 这里应该重写写一下未初始化的逻辑
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

    // 获取需要处理的输入数据
    std::string partial_output = cmd.input_data;
    Batch bt;
    model_->forward(bt);
    // TODO：进行计算

    // 最后一个设备处理完了 通知 Engine
    if (cmd.remaining_workers->fetch_sub(1, std::memory_order_acq_rel) == 1)
    {
        std::cout << "[Worker " << get_id() << " TID: " << get_thread_id_str() << "] (ReqID: " << cmd.request_id << ") This is the LAST worker for this INFER task. Setting master promise." << std::endl;
        Result final_res;
        final_res.request_id = cmd.request_id;
        final_res.success = true;
        // TODO：这里之后稍微改改
        std::reverse(partial_output.begin(), partial_output.end());
        partial_output = "[W0_reversed] " + partial_output;
        final_res.output_data = "Aggregated (simulated by last worker " + std::to_string(get_id()) + "): " + partial_output;
        cmd.master_promise->set_value(final_res);
    }
    else
    {
        std::cout << "[Worker " << get_id() << " TID: " << get_thread_id_str() << "] (ReqID: " << cmd.request_id << ") Not the last worker for this INFER task. (" << cmd.remaining_workers->load() << " remaining)" << std::endl;
    }
}
