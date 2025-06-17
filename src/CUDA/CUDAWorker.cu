#include "CUDA/CUDAWorker.h"
#include "CUDA/CUDAUtils.h"


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

Result CUDAWorker::handle_init()
{
    CUDA_CHECK(cudaSetDevice(get_id()));
    is_initialized_ = true;

    return Result{0, true, "", ""};
}

Result CUDAWorker::handle_infer(std::shared_ptr<Batch> batch)
{
    std::cout << "[Worker " << get_id() << "] Handling INFER command..." << std::endl;
    std::vector<std::vector<int>> token_batch = batch->token_batch;
    for(int i = 0; i < token_batch.size(); ++i)
    {
        std::cout << "Batch " << i << ": ";
        for(int token : token_batch[i])
        {
            std::cout << token << " ";
        }
        std::cout << std::endl;
    }
    // 获取需要处理的输入数据
    // std::string partial_output = cmd.input_data;
    // Batch bt;
    // model_->forward(bt);
    // TODO：进行计算 存到 result 中
    return Result{0, true, "", ""};
}
