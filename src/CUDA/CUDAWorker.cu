#include "CUDA/CUDAWorker.h"
#include "CUDA/CUDAUtils.h"

void CUDAWorker::handle_init(Command cmd)
{
    std::cout << "[Worker " << get_id() << " TID: " << get_thread_id_str() << "] Handling INIT command..." << std::endl;

    CUDA_CHECK(cudaSetDevice(get_id()));

    std::cout << "Worker " + std::to_string(get_id()) + " initialized successfully.";

    void *d_array = memory_manager_.allocate(1000000 * sizeof(float));
    
    std::this_thread::sleep_for(std::chrono::seconds(500));

    try
    {
        Result res;
        res.success = true;
        cmd.individual_promise.set_value(res);
    }
    catch (const std::future_error &e)
    {
        std::cerr << "[Worker " << get_id() << " TID: " << get_thread_id_str()
                  << "] Future error setting value for INIT promise: " << e.what() << std::endl;
    }
}