#ifndef PARAMETER_H
#define PARAMETER_H

#include "Variable.h"

class Parameter : public Variable {
public:
    // 构造函数
    Parameter(const std::string& _name, const size_t _eleNum, const size_t _elemLen, const std::string& _device,
                                            bool _malloc_mem = false) : Variable(_eleNum, _elemLen, _device, _name) {
        if(_malloc_mem) {
            // std::cout << " this is param  name is _ " << _name << std::endl;
            value = manager.allocateShared(size, device, _name);
        }
    }

    // 虚析构函数
    ~Parameter() override { value.reset(); }

    void setShared(){ shared = true; }
    bool Share() { return shared; }

    // 给 cache 准备的
    void copy(float* from, const size_t length, const size_t from_offset, const size_t to_offset) {
        manager.copy(from + from_offset, value.get() + to_offset, length, device);
    }

    void to(const std::string& new_dev) override {
        if (new_dev == device) return;

        if(new_dev == "")
            throw std::logic_error("there is no device " + new_dev);
        
        if(shared) {
            name.replace(0, device.length(), new_dev);
            if(manager.FindMem(name)) {
                value = manager.GetMem(name);
            } else {
                std::shared_ptr<float []> val = manager.deepCopy(value, size, device);
                manager.toDevice(val, size, device, new_dev, name);
                manager.RegisteMem(name, val);
                value = val;
            }
        }else{
            manager.toDevice(value, size, device, new_dev, name);
        }

        device = new_dev;
    }
private:
    bool shared = false;
};

#endif