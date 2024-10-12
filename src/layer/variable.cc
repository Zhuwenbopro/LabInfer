#include "Variable.h"
#include "Manager.h"  // 在这里包含 Manager.h，因为我们需要使用它的定义

// 在 C++ 中，当创建一个子类（派生类）的对象时，父类（基类）的构造函数会先于子类的构造函数被调用。
Variable::Variable(const std::string& _name, const std::vector<size_t>& _shape, const std::string& _device, 
    bool _malloc_mem) : shape(_shape), size(1), name(_name), device(_device) {

    for (const auto& dim : shape) {
        size *= dim;
    }

    if(_malloc_mem) {
        Manager& manager = Manager::getInstance();
        value = manager.allocate(size, device);
    }
}

void Variable::to(const std::string& new_dev) {
    if (new_dev == device) return;
    if(new_dev == "") return;
    
    Manager& manager = Manager::getInstance();
    manager.toDevice(value, size, device, new_dev);
    device = new_dev;
}

Variable::~Variable() {
    value.reset();
}

// 深拷贝
void Variable::_copy(const Variable& from) {
    Manager& manager = Manager::getInstance();
    value = manager.deepCopy(from.sharedPtr(), size, device);
}