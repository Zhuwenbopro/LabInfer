#ifndef DEVICE_H
#define DEVICE_H

#include <iostream>
#include <string>

class Device {
public:
    // 构造函数和析构函数
    Device(const std::string& deviceName = "cpu");
    Device(const Device& other);
    ~Device();

    // 重载 << 运算符
    friend std::ostream& operator<<(std::ostream& os, const Device& obj);

private:
    // 成员变量
    std::string dev;    // 设备名称
    int nodeId;         // 设备结点号
    int deviceId;       // 设备号
};

#endif // DEVICE_H
