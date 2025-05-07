#pragma once
#include <string>
#include <iostream>
#include "Tensor.h"
#include "Parameter.h"
#include "ParamLoader.h"
#include "Device.h"
#include "Communicator.h"

class Layer
{
public:
    Layer(const std::string& name = "layer") : name_(name)
    {
    }

    ~Layer()
    {
    }

    void setDevice(Device* device)
    {
        device_ = device;
    }

    void setCommunicator(Communicator* communicator)
    {
        communicator_ = communicator;
    }


    void forward(Tensor &input, Tensor &output)
    {
        // for(int i = 0; i < n_; ++i)
        // {
        //     output[i] = input[i] + param_[i];
        // }
    }

    void print()
    {
        // 打印参数
        // std::cout << "Layer " << name_ << " weights: ";
        // for (int i = 0; i < n_; ++i)
        // {
        //     std::cout << param_[i] << " ";
        // }
        // std::cout << "\n";
    }
    
private:
    std::string name_;
    int n_;
    Device* device_;
    Communicator* communicator_;
};