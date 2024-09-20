#include "Device.h"
#include <iostream>

int main() {
    // 创建 Device 对象
    Device device("MyDevice");

    // 使用重载的 << 运算符输出设备信息
    std::cout << device << std::endl;

    return 0;
}

