#include "Engine.h"

void Engine::add_request(std::shared_ptr<Request> req) {
    std::lock_guard<std::mutex> lock(pending_request_mtx_);
    pending_requests_.push(req);
    pending_request_cv_.notify_one();
}

