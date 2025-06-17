#include "Engine.h"
#include "common.h"

int main()
{
    try
    {
        const int NUM_WORKERS = 2; // Try with 1, 2, or more
        Engine engine(NUM_WORKERS, CUDA); // Change to CPU or CUDA as needed

        engine.initialize_workers();

        std::cout << "\n--- Submitting Inference Tasks (will be sent to all workers) ---" << std::endl;
        std::vector<std::future<Result>> inference_futures;

        struct Batch batch;
        batch.request_ids = {sole::uuid4(), sole::uuid4(), sole::uuid4()};
        batch.token_batch = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        batch.position_batch = {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}};
        batch.batch_shape = {3, 3, 3};
        
        inference_futures.push_back(engine.submit_inference_request(batch));
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Stagger submissions a bit

        std::cout << "\n--- Waiting for Inference Results (master promise from last worker) ---" << std::endl;
        for (size_t i = 0; i < inference_futures.size(); ++i)
        {
            try
            {
                Result res = inference_futures[i].get();
                if (res.success)
                {
                    std::cout << "[Main] Received FINAL result for ReqID " << res.request_id
                              << ": '" << res.output_data << "'" << std::endl;
                }
                else
                {
                    std::cerr << "[Main] Error for FINAL ReqID " << res.request_id
                              << ": " << res.error_message << std::endl;
                }
            }
            catch (const std::exception &e)
            {
                std::cerr << "[Main] Exception getting result for request " << i << ": " << e.what() << std::endl;
            }
        }
        std::cout << "\n--- Engine and Workers will be shut down by Engine's destructor ---" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Unhandled exception in main: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
