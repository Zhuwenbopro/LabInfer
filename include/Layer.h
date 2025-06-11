#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include "Function.h"
#include "Batch.h"

class Bacth;

class Layer {
protected:
    std::unordered_map<std::string, std::shared_ptr<void>> params_;
    std::unordered_map<std::string, Layer*> layers_;
    std::string name_;
    Function* function_;
    int TP;
public:
    Layer() { }
    virtual ~Layer() = default;

    // 禁止拷贝构造和拷贝赋值，避免浅拷贝问题
    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;

    virtual void forward(Bacth& batch) = 0;


    // TODO : for the further further future
    // virtual void backward(std::shared_ptr<void> x) = 0;
};
