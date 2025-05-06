#pragma once
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

class SafePrinter {
public:
    // 可传入任意类型、任意数量的参数
    template<typename... Args>
    static void print(Args&&... args) {
        std::lock_guard<std::mutex> lk(getMutex());
        // 把所有参数拼到一个 stringstream 里
        (getStream() << ... << std::forward<Args>(args));
        // 输出并换行
        std::cout << getStream().str() << std::endl;
        // 清空 buffer，准备下次使用
        getStream().str("");
        getStream().clear();
    }

private:
    // 单例的互斥量
    static std::mutex& getMutex() {
        static std::mutex mtx;
        return mtx;
    }
    // 线程局部的缓存流（减少每次申请销毁开销）
    static std::ostringstream& getStream() {
        thread_local std::ostringstream oss;
        return oss;
    }
};
