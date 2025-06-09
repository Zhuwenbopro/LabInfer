#pragma once

#include <memory>
#include <queue>
#include <vector>
#include <mutex>
#include <condition_variable>
#include "common.h"
#include "Worker.h"


class OutputToken
{
};

class Scheduler
{
};

class Engine
{
private:
    std::queue<std::shared_ptr<Request>> pending_requests_;
    std::mutex pending_request_mtx_;
    std::condition_variable pending_request_cv_;

    std::vector<std::shared_ptr<RequestState>> active_request_states_;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::unique_ptr<Scheduler> scheduler_;
    // std::unique_ptr<InterNodePipelineCommunicator> inter_node_comm_;
public:
    // 外部添加请求
    void add_request(std::shared_ptr<Request> req);
    // 外部获取生成的 token
    bool get_output_token(OutputToken &token);
    // 启动 Engine 主循环
    void run();
    // 停止 Engine
    void stop();

private:
    // 主处理逻辑循环
    void process_loop();
    // 给 Workers 发信号并收集结果
    ModelOutputBatch dispatch_to_workers(const ModelInputBatch &input_batch);
};