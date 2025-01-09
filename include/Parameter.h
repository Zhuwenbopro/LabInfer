#ifndef PARAMETER_H
#define PARAMETER_H

#include "Variable.h"

template <typename T>
class Parameter : public Variable<T> {
public:
    // 构造函数
    Parameter(
        const size_t _eleNum = 0, 
        const size_t _elemLen = 0, 
        const std::string& _device = "cpu", 
        const std::string& _name = "name",
        bool _automalloc = false
    ) : Variable<T>(_eleNum, _elemLen, _device, _name, _automalloc) { }

    // 虚析构函数
    ~Parameter() override {
        if(shared) {
            if(this->value.use_count() == 2) {
                this->deviceManager.FreeMem(this->name);
            }
        }
    }

    // 不可逆的操作
    void setShared(){
        if(shared) return;

        shared = true;
        this->name = this->device + ":" + this->name;
        if(this->deviceManager.FindMem(this->name)) {
            this->value = this->deviceManager.GetMem(this->name);
        } else {
            this->deviceManager.RegisteMem(this->name, this->value);
        }
    }

    bool Share() { return shared; }

    void to(const std::string& _device) override {
        if (_device == this->device) return;

        if(_device == "")
            throw std::logic_error("there is no device " + _device);
        
        if(shared) {
            this->name.replace(0, this->device.length(), _device);

            if(this->deviceManager.FindMem(this->name)) {
                this->value = this->deviceManager.GetMem(this->name);
            } else {
                if(this->value.use_count() == 2)
                    this->deviceManager.FreeMem(this->name);
                this->deviceManager.toDevice(this->value, this->Bytes(), this->device, _device);
                this->deviceManager.RegisteMem(this->name, this->value);
            }
        }else{
            this->deviceManager.toDevice(this->value, this->Bytes(), this->device, _device);
        }

        this->device = _device;
    }

private:
    bool shared = false;
};

#endif