#ifndef WORKER_H
#define WORKER_H

#include "SafeQueue.h"
#include <thread>
#include <atomic>
#include <iostream>
#include "layers/layers.h"

#define FIRST 0x1
#define LAST 0x2

class Worker {
public:
    // 构造函数启动线程
    Worker(const std::string& _name, 
            std::shared_ptr<SafeQueue<InputWarp>> inQueue, 
            std::shared_ptr<SafeQueue<InputWarp>> outQueue,
            Layer* _layer
            ) : name(_name), queue_in(inQueue), queue_out(outQueue), layer(_layer), running(false) {
        run();
    }

    // 禁止拷贝和赋值操作（因为 std::thread 无法拷贝）
    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;

    // 析构函数通知线程退出并等待
    ~Worker() {
        if(running) this->stop();
        delete layer;
    }

    // FIXME : can not quit correctly
    void stop() {
        if(!running) return;
        stopFlag_.store(true); // 设置退出标志
        if (workerThread_.joinable()) {
            workerThread_.join(); // 等待线程退出
        }
        running = false;
        std::cout << "thread[" << name << "] stopped" << std::endl;
    }

private:
    std::string name;
    std::shared_ptr<SafeQueue<InputWarp>> queue_in;
    std::shared_ptr<SafeQueue<InputWarp>> queue_out;
    Layer* layer;

    std::thread workerThread_;       // 工作线程
    bool running;
    std::atomic<bool> stopFlag_{false}; // 标志线程是否退出

    void run() {
        if(running) return;
        workerThread_ = std::thread(&Worker::threadFunction, this);
        running = true;
    }

    // 线程的工作函数
    void threadFunction() {
        while (!stopFlag_.load()) {
            InputWarp inputWarp = queue_in->pop();
            layer->forward(inputWarp);
            queue_out->push(inputWarp);
        }
    }
};

#endif