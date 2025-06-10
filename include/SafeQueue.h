// SafeQueue.h
#include <queue>
#include <condition_variable>
#include <mutex>

template <typename T>
class SafeQueue
{
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cond_var_;

public:
    void push(T value)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(value));
        cond_var_.notify_one();
    }

    T pop()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_var_.wait(lock, [this]
                       { return !queue_.empty(); });
        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }
    // ... (try_pop, empty if needed, not used in this version of demo)
};
