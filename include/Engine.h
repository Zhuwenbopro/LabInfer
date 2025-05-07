#pragma once
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
#include "Worker.h"

class Engine
{
public:
    explicit Engine(int n, int world_id = 0) : world_size(n), counter(n), world_id(world_id) {

        threads.reserve(world_size);
        workers.reserve(world_size);

        for (int i = 0; i < world_size; ++i) {
            // emplace_back(args…) constructs the Worker in‑place inside the vector
            // using the constructor arguments you provide, avoiding an extra copy or move.
            workers.emplace_back(world_id, i, counter, mtx, cv);
            // launching a thread whose entry‑point is operator() on the very last Worker object,
            // and by using std::ref to ensure it calls the real Worker (not a temporary copy).
            threads.emplace_back(std::ref(workers.back()));
        }

        // 主线程等待所有 Worker 完成初始化
        {
            // As soon as lk is constructed, it calls mtx.lock(), so you hold exclusive ownership of the Engine’s mtx mutex.
            std::unique_lock<std::mutex> lk(mtx);
            // Atomically unlock mtx and suspend the thread while waiting.
            // When notified (and the predicate returns true), it will re‑lock mtx before returning.
            cv.wait(lk, [this] { return counter.load() == 0; });
        }
        // std::cout << "所有线程都完成操作，主线程继续。\n";
    }

    ~Engine() {
        
        stop();

        for (auto &t : threads) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    // TODO 5.9 完成数据流动验证
    void step() {
        
    }

    // TODO 应该是发送某一个特殊的数据信号，通知网络中的所有结点停止工作
    void stop() {

    }

private:
    const int world_size, world_id;
    std::atomic<int> counter;
    std::mutex mtx;
    std::condition_variable cv;

    std::vector<Worker> workers;
    std::vector<std::thread> threads;
};