#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <ctime>
#include <random>

const int MAX_RUNTIME_SECONDS = 60;

class Consumer {
public:
    void consume() {
        while (true) {
            std::unique_lock<std::mutex> lock(queueMutex);
            cv.wait(lock, [this]{ return !dataQueue.empty() || done; });

            if (done && dataQueue.empty()) {
                break;
            }

            while (!dataQueue.empty()) {
                int data = dataQueue.front();
                dataQueue.pop();
                lock.unlock(); // Unlock before processing to allow producer to add more items

                auto now = std::time(nullptr);
                char buf[100];
                std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
                std::cout << "Consumed: " << data << " at " << buf << std::endl;

                lock.lock(); // Re-lock after processing
            }
        }
    }

    void add(int data) {
        std::lock_guard<std::mutex> lock(queueMutex);
        dataQueue.push(data);
        cv.notify_one();
    }

    void setDone(bool d) {
        std::lock_guard<std::mutex> lock(queueMutex);
        done = d;
        cv.notify_all();
    }

private:
    std::queue<int> dataQueue;
    std::mutex queueMutex;
    std::condition_variable cv;
    bool done = false;
};

class Producer {
public:
    Producer(Consumer& consumer) : consumer(consumer) {}

    void produce() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(2, 5); // Random time between 2 and 5 seconds

        auto startTime = std::chrono::steady_clock::now();
        while (true) {
            auto currentTime = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime).count() >= MAX_RUNTIME_SECONDS) {
                break;
            }

            int data = rand() % 100; // Generate a random integer
            std::this_thread::sleep_for(std::chrono::seconds(dis(gen))); // Sleep for a random duration

            consumer.add(data);
            std::cout << "Produced: " << data << std::endl;
        }
    }

private:
    Consumer& consumer;
};

int main() {
    std::cout << "系统启动" << std::endl;

    Consumer consumer;
    Producer producer(consumer);

    std::thread prodThread(&Producer::produce, &producer);
    std::thread consThread(&Consumer::consume, &consumer);

    std::this_thread::sleep_for(std::chrono::seconds(MAX_RUNTIME_SECONDS));

    consumer.setDone(true); // Set the done flag in the Consumer class

    prodThread.join();
    consThread.join();

    std::cout << "系统关闭" << std::endl;
    return 0;
}



