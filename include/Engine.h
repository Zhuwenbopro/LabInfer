#pragma once
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
#include "Worker.h"


class Engine {
    public:
        explicit Engine(int n) : N(n) , counter(n)
        {
            threads.reserve(N);
            workers.reserve(N);
            for (int i = 0; i < N; ++i) {
                // emplace_back(args…) constructs the Worker in‑place inside the vector 
                // using the constructor arguments you provide, avoiding an extra copy or move.
                workers.emplace_back(i, counter, mtx, cv);
                threads.emplace_back(std::ref(workers.back()));
            }

            // 主线程等待所有 Worker 完成
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv.wait(lk, [this] { return counter.load() == 0; });
            }
            std::cout << "所有线程都完成操作，主线程继续。\n";
        }

        ~Engine() {
            for (auto &thread : threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
        }
    
        void step() {
        }
    
    private:
        const int                          N;
        std::atomic<int>                   counter;
        std::mutex                         mtx;
        std::condition_variable            cv;
        std::vector<Worker>                workers;
        std::vector<std::thread>           threads;
    };