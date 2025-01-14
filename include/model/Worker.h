#ifndef WORKER_H
#define WORKER_H

#include "SafeQueue.h"
#include <thread>
#include <atomic>
#include <iostream>

#define FIRST 0x1
#define LAST 0x2

class Worker {
public:
    // 构造函数启动线程
    Worker(const std::string& _name, 
            std::shared_ptr<SafeQueue<std::string>> inQueue, 
            std::shared_ptr<SafeQueue<std::string>> outQueue,
            uint32_t _state = 0
            ) : name(_name), queue_in(inQueue), queue_out(outQueue), state(_state) {
        std::cout << "start: " << name << std::endl;
        workerThread_ = std::thread(&Worker::threadFunction, this);
    }

    // 禁止拷贝和赋值操作（因为 std::thread 无法拷贝）
    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;

    // 析构函数通知线程退出并等待
    ~Worker() {
        stopFlag_.store(true); // 设置退出标志
        if (workerThread_.joinable()) {
            workerThread_.join(); // 等待线程退出
        }
    }

    void run() {}

    void stop() {}
private:
    std::string name;
    std::shared_ptr<SafeQueue<std::string>> queue_in;
    std::shared_ptr<SafeQueue<std::string>> queue_out;
    uint32_t state;

    std::thread workerThread_;       // 工作线程
    std::atomic<bool> stopFlag_{false}; // 标志线程是否退出

    // 线程的工作函数
    void threadFunction() {
        std::string message;
        while (!stopFlag_.load()) {
            // 进行 merge 把已经积累起来的合并
            if(state & FIRST) {
                printMsg(name+" is getting message...");
                message = queue_in->mergepop();
                printMsg(name + " : " + message);
            } else {
                printMsg(name+" is getting message...");
                message = queue_in->pop();
                printMsg(name + " : " + message);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(500)); // 模拟工作
            queue_out->push(message);
            printMsg(name + " push msg : " + message);

            // 判断是否结束词
            if(state & LAST) {
                
            }
        }
        std::cout << "Thread exiting..." << std::endl;
    }

    void printMsg(const std::string& msg) {
        static std::mutex print_mutex;
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cout << msg << std::endl;
    }

};

#endif