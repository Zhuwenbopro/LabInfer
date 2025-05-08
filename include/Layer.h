#pragma once
#include <string>
#include "Device.h"
#include "Communicator.h"
#include <stdexcept>


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

    void loadParam()
    {

    }

    void forward(float *input, float *output)
    {
        if(!input || !output)
        {
            throw std::invalid_argument("Input or output tensor is null");
        }
        
        output[0] = input[0] * 2;
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
    Device* device_;
    Communicator* communicator_;
};