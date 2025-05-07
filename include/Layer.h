#pragma once
#include <string>
#include "Tensor.h"
#include "Parameter.h"
#include "ParamLoader.h"

class Layer
{
public:
    Layer(std::string& name, int n)
        : name_(name), n_(n), param_(n_)
    {
    }

    ~Layer()
    {
    }

    void forward(Tensor &input, Tensor &output)
    {
        for(int i = 0; i < n_; ++i)
        {
            output[i] = input[i] + param_[i];
        }
    }

    void print()
    {
        // 打印参数
        std::cout << "Layer " << name_ << " weights: ";
        for (int i = 0; i < n_; ++i)
        {
            std::cout << param_[i] << " ";
        }
        std::cout << "\n";
    }
    
private:
    std::string name_;
    int n_;
    Parameter param_;
};