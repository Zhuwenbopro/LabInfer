#pragma once
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
#include "Worker.h"


class Engine {
    public:
        explicit Engine(int n)
          : N(n)
          , counter(n)
        {}
    
        void run() {
            // 构造 N 个 Worker，并启动对应线程
            std::vector<std::thread> threads;
            threads.reserve(N);
            workers.reserve(N);
            for (int i = 0; i < N; ++i) {
                workers.emplace_back(i, counter, mtx, cv);
                threads.emplace_back(std::ref(workers.back()));
            }
    
            // 主线程等待所有 Worker 完成
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv.wait(lk, [this] { return counter.load() == 0; });
            }
            std::cout << "所有线程都完成操作，主线程继续。\n";
    
            // 回收线程
            for (auto &t : threads) {
                if (t.joinable())
                    t.join();
            }
        }
    
    private:
        const int                          N;
        std::atomic<int>                   counter;
        std::mutex                         mtx;
        std::condition_variable            cv;
        std::vector<Worker>                workers;
    };