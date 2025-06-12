#pragma once

#include "layer.h"
#include "function.h"

class Linear : public Layer {
public:

    Linear(LinearFuncPtr func_ptr) : func_ptr_(func_ptr)
    {
        name_ = "Linear";
    }

    void forward(Batch& batch) override {
        func_ptr_(nullptr, nullptr, nullptr, 0, 0, 0); // TODO: Implement the actual forward logic
    }
private:
    LinearFuncPtr func_ptr_;
};