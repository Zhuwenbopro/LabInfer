#ifndef SAFEQUEUE_H
#define SAFEQUEUE_H

#include <memory>
#include <iostream>
#include <queue>
#include <mutex>
#include <semaphore> // 使用 C++20 的 counting_semaphore

template <typename T>
class SafeQueue {
private:
    std::string name_;
    std::queue<T> queue_;
    std::mutex mutex_;
    std::counting_semaphore<> sem_; // 使用 C++20 的 counting_semaphore

public:
    // 构造函数，初始化信号量计数为 0
    SafeQueue(const std::string& name = "") 
        : sem_(0), name_(name) {}

    // 合并队列中的元素
    T mergepop() {
        sem_.acquire(); // 等待信号量大于 0，即队列中有元素
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << "queue " << name_ << " merge pop in, queue.size = " << queue_.size() << std::endl;
        T merged_value = queue_.front();
        queue_.pop();

        while (queue_.size() > 0) {
            sem_.acquire();
            merged_value += queue_.front(); // 合并操作，假设 T 支持 +=
            queue_.pop();
            std::cout << "queue " << name_ << " cycle queue.size = " << queue_.size() << std::endl;
        }

        std::cout << "queue " << name_ << " merge pop out, queue.size = " << queue_.size() << std::endl;
        return merged_value;
    }

    // 将元素推入队列
    void push(const T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(value);
        sem_.release(); // 发出信号，表示队列中有新元素
    }

    // 从队列中取出一个元素
    T pop() {
        sem_.acquire(); // 等待信号量大于 0，即队列中有元素
        std::lock_guard<std::mutex> lock(mutex_);

        std::cout << "queue " << name_ << " pop in" << std::endl;
        T value = queue_.front();
        queue_.pop();
        std::cout << "queue " << name_ << " pop out" << std::endl;
        return value;
    }
};

#endif