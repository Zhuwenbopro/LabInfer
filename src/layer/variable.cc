#include "Variable.h"
#include "Manager.h"  // 在这里包含 Manager.h，因为我们需要使用它的定义

void Variable::to(const std::string& new_dev) {
    if (new_dev == device) return;

    Manager& manager = Manager::getInstance();
    manager.toDevice(*this, new_dev);
}