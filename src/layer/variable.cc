#include "Variable.h"
#include "Manager.h"  // 在这里包含 Manager.h，因为我们需要使用它的定义

Variable::Variable(const std::string& _name, float* _value, const std::vector<size_t>& _shape, 
        const std::string& _device, bool _malloc_mem) : value(_value), shape(_shape), size(1), name(_name), device(_device) {

        for (const auto& dim : shape) {
            size *= dim;
        }

        if(_malloc_mem) {
            Manager& manager = Manager::getInstance();
            value = manager.getDevice(device).allocate(size);
        }
    }

void Variable::to(const std::string& new_dev) {
    if (new_dev == device) return;
    
    Manager& manager = Manager::getInstance();
    manager.toDevice(*this, new_dev);
}

Variable::~Variable() {
    // std::cout << "variable " << name << " release" << std::endl;
    Manager& manager = Manager::getInstance();
    manager.getDevice(device).deallocate(value);
}

float* Variable::_copy() const {
    Manager& manager = Manager::getInstance();
    float* ptr =  manager.getDevice(device).allocate(size);
    manager.getDevice(device).copy(value, ptr, size);
    return ptr;
}