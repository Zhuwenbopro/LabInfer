#include "Variable.h"

void Variable::to(const std::string& new_dev) {
    if (new_dev == device) return;
    if(new_dev == "")
        throw std::logic_error("there is no device " + new_dev);
    
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
