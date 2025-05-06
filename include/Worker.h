#pragma once
#include <atomic>
#include <mutex>
#include <condition_variable>
#include "Device.h"
#include "Layer.h"
#include "ParamLoader.h"
#include "SafePrinter.h"

class Worker
{
public:
    Worker(int id,
           std::atomic<int> &counter,
           std::mutex &mtx,
           std::condition_variable &cv)
        : id_(id), counter_(counter), mtx_(mtx), cv_(cv)
    {
    }

    // 重载 operator()，让 std::thread 可以直接用 Worker 对象启动
    void operator()()
    {
        // … 更复杂的操作都可以写在这里 …
        SafePrinter::print("线程 ", id_, " 完成操作");

        // 原子地把 counter 减 1；如果是最后一个，就通知 Engine
        if (--counter_ == 0)
        {
            std::lock_guard<std::mutex> lk(mtx_);
            cv_.notify_one();
        }

        SafePrinter::print("还有 ", counter_.load(), " 个信号");
    }

private:
    int id_;
    std::atomic<int> &counter_;
    std::mutex &mtx_;
    std::condition_variable &cv_;
};