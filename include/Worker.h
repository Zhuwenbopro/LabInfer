#pragma once
// #include "Engine.h"
#include "SafeQueue.h"
#include "common.h"
#include <thread>
#include <atomic>
#include <algorithm>
#include <sstream>
#include <iomanip>

class Engine;
class Worker
{
public:
    Worker(int id, Engine *engine) : id_(id), engine_(engine), running_(false)
    {
        std::cout << "[Worker " << get_id() << "] Created." << std::endl;
    }

    ~Worker()
    {
        stop();
        std::cout << "[Worker " << get_id() << "] Destroyed." << std::endl;
    }

    void start();

    void stop();

    void push_command(Command cmd);

    int get_id() const { return id_; }

    // Helper to get a nicely formatted thread ID string
    std::string get_thread_id_str();

private:
    void process_loop();
    

    virtual void handle_init(Command cmd);

    void handle_infer(Command cmd);

    int id_;
    std::atomic<bool> initialized_;
    
    Engine *engine_;
    SafeQueue<Command> command_queue_;
    std::thread thread_;
    std::atomic<bool> running_;
    bool is_initialized_ = false;
    std::string model_name_;
};
