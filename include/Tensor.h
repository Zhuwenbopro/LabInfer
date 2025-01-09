#ifndef TENSOR_H
#define TENSOR_H

#include "Variable.h"

template <typename T>
class Tensor : public Variable<T> {
private:
public:
    Tensor(
        const size_t _eleNum = 0, 
        const size_t _elemLen = 0, 
        const std::string& _device = "cpu", 
        const std::string& _name = "name"
    ) : Variable<T>(_eleNum, _elemLen, _device, _name) { }

    Tensor(const Tensor<T>& other) 
    : Variable<T>(
        other.ElemNum(), 
        other.ElemLen(), 
        other.Device(), 
        other.Name() + "_copy"
    ) {
        this->copy(0, other, 0, this->Size());
    }

    ~Tensor() override { }

    void to(const std::string& _device) override {
        if (_device == this->device) return; // 使用 this-> 来明确访问基类成员
        
        if(_device == "") 
            throw std::logic_error("there is no device " + _device);

        if(this->value != nullptr)
            this->deviceManager.toDevice(this->value, this->Bytes(), this->device, _device);
            
        this->device = _device;
    }
};

#endif