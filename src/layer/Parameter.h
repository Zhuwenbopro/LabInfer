#ifndef PARAMETER_H
#define PARAMETER_H

#include "Variable.h"

class Parameter : public Variable {
public:
    // 构造函数
    Parameter(const std::string& _name, const std::vector<size_t>& _shape, 
        const std::string& _device, bool _malloc_mem = false) : Variable(_name, _shape, _device) {
        if(shape.size() > 2 && shape[0] != 1) {
            throw std::logic_error("this is not a param, but a tensor" + _name);
        }

        for (const auto& dim : shape) {
            size *= dim;
        }

        if(_malloc_mem) {
            Manager& manager = Manager::getInstance();
            value = manager.allocate(size, device);
        }
    }

    // 虚析构函数
    ~Parameter() override { }

    void setShared(){ shared = true; }
    bool Share() { return shared; }

    void to(const std::string& new_dev) override {
        if (new_dev == device) return;
        if(new_dev == "")
            throw std::logic_error("there is no device " + new_dev);
        
        Manager& manager = Manager::getInstance();

        if(shared) {
            name.replace(0, device.length(), new_dev);
            if(manager.FindMem(name)) {
                value = manager.GetMem(name);
            } else {
                std::shared_ptr<float []> val = manager.deepCopy(value, size, device);
                manager.toDevice(val, size, device, new_dev);
                manager.RegisteMem(name, val);
                value = val;
            }
        }else{
            manager.toDevice(value, size, device, new_dev);
        }

        device = new_dev;
    }

private:
    bool shared = false;
};

#endif