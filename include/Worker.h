#pragma once
// #include "Engine.h"
#include "SafeQueue.h"
#include "common.h"
#include <thread>
#include <atomic>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include "layers/layer.h"
#include "layers/Linear.h"
#include "function.h"
#include "registry.h"

class Engine;
class Worker
{
public:
    // 初始化的时候把模型分配好
    Worker(int id, Engine *engine);

    ~Worker();

    void start();

    void stop();

    void push_command(Command cmd);

    int get_id() const { return id_; }

    // Helper to get a nicely formatted thread ID string
    std::string get_thread_id_str();

protected:
    void process_loop();
    

    virtual void handle_init(Command cmd);
    void handle_infer(Command cmd);

    int id_;
    std::atomic<bool> initialized_;
    
    Engine *engine_;
    std::unique_ptr<Layer> model_;

    SafeQueue<Command> command_queue_;
    std::thread thread_;
    std::atomic<bool> running_;
    bool is_initialized_ = false;
    std::string model_name_;
};
