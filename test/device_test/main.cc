#include "DeviceFactory.h"
#include <iostream>

int main() {

    Device *d1 = DeviceFactory::getDevice("cpu");
    Device *d2 = DeviceFactory::getDevice("cpu");

    // 使用重载的 << 运算符输出设备信息
    std::cout << d1 << " : " << d1->getDeviceName() << std::endl;
    std::cout << d2 << " : " << d2->getDeviceName() << std::endl;

    return 0;
}

