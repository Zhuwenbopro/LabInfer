#pragma once

#include <memory>
#include <string>
#include <unordered_map>

class Batch{};

class Layer {
protected:
    std::unordered_map<std::string, std::shared_ptr<void>> params_;
    std::unordered_map<std::string, Layer*> layers_;
    std::string name_;
    int TP;
public:
    Layer() { }
    virtual ~Layer() = default;

    // TODO
    void load_params(const std::unordered_map<std::string, std::shared_ptr<void>>& params) {
        
    }

    virtual void forward(Batch& batch) = 0;

    // TODO : for the further further future
    // virtual void backward(std::shared_ptr<void> x) = 0;
};
