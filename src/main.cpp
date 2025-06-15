/*
#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <ctime>
#include <random>

const int MAX_RUNTIME_SECONDS = 60;

class Consumer {
public:
    void consume() {
        while (true) {
            std::unique_lock<std::mutex> lock(queueMutex);
            cv.wait(lock, [this]{ return !dataQueue.empty() || done; });

            if (done && dataQueue.empty()) {
                break;
            }

            while (!dataQueue.empty()) {
                int data = dataQueue.front();
                dataQueue.pop();
                lock.unlock(); // Unlock before processing to allow producer to add more items

                auto now = std::time(nullptr);
                char buf[100];
                std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
                std::cout << "Consumed: " << data << " at " << buf << std::endl;

                lock.lock(); // Re-lock after processing
            }
        }
    }

    void add(int data) {
        std::lock_guard<std::mutex> lock(queueMutex);
        dataQueue.push(data);
        cv.notify_one();
    }

    void setDone(bool d) {
        std::lock_guard<std::mutex> lock(queueMutex);
        done = d;
        cv.notify_all();
    }

private:
    std::queue<int> dataQueue;
    std::mutex queueMutex;
    std::condition_variable cv;
    bool done = false;
};

class Producer {
public:
    Producer(Consumer& consumer) : consumer(consumer) {}

    void produce() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(2, 5); // Random time between 2 and 5 seconds

        auto startTime = std::chrono::steady_clock::now();
        while (true) {
            auto currentTime = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime).count() >= MAX_RUNTIME_SECONDS) {
                break;
            }

            int data = rand() % 100; // Generate a random integer
            std::this_thread::sleep_for(std::chrono::seconds(dis(gen))); // Sleep for a random duration

            consumer.add(data);
            std::cout << "Produced: " << data << std::endl;
        }
    }

private:
    Consumer& consumer;
};

int main() {
    std::cout << "系统启动" << std::endl;

    Consumer consumer;
    Producer producer(consumer);

    std::thread prodThread(&Producer::produce, &producer);
    std::thread consThread(&Consumer::consume, &consumer);

    std::this_thread::sleep_for(std::chrono::seconds(MAX_RUNTIME_SECONDS));

    consumer.setDone(true); // Set the done flag in the Consumer class

    prodThread.join();
    consThread.join();

    std::cout << "系统关闭" << std::endl;
    return 0;
}

*/

#include "Engine.h"
// #include "Worker.h"
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

        inference_futures.push_back(engine.submit_inference_request("Hello Distributed World"));
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Stagger submissions a bit
        inference_futures.push_back(engine.submit_inference_request("C++ Tensor Parallelism Demo"));

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
