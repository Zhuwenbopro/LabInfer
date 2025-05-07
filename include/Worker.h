#pragma once
#include <atomic>
#include <mutex>
#include <condition_variable>
#include "Device.h"
#include "Layer.h"
#include "Communicator.h"
#include "ParamLoader.h"
#include "SafePrinter.h"

class Worker
{
public:
    Worker(int world, int rank,
           std::atomic<int> &counter,
           std::mutex &mtx,
           std::condition_variable &cv)
        : world_(world), rank_(rank), counter_(counter), mtx_(mtx), cv_(cv)
    {
        device_ = new Device(rank_);
        communicator_ = new Communicator();
        layer_ = new Layer("layer");
    }

    // 重载 operator()，让 std::thread 可以直接用 Worker 对象启动
    void operator()()
    {
        init();

        run();

    }

private:
    int world_;
    int rank_;
    std::atomic<int> &counter_;
    std::mutex &mtx_;
    std::condition_variable &cv_;

    Device* device_;
    Layer* layer_;
    Communicator* communicator_;

    void init() 
    {
        device_->init();
        communicator_->init();

        layer_->setDevice(device_);
        layer_->setCommunicator(communicator_);


        // TODO init device 5.7
        // TODO init layer 5.7
        // Load parameters from file into layer
        // TODO init Communicator 5.7

        SafePrinter::print("world ", world_, " rank ", rank_, " finished init");

        if (--counter_ == 0)
        {
            // As soon as lk is constructed, it calls mtx.lock(), so you hold exclusive ownership of the Engine’s mtx mutex.
            std::lock_guard<std::mutex> lk(mtx_);
            cv_.notify_one();
        }
    }

    void run()
    {

    }
};