// Device.h
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

    // 获取设备名称
    std::string getDeviceName() const;

private:
    // 成员变量
    std::string dev;    // 设备名称
    int nodeId;         // 设备结点号
    int deviceId;       // 设备号
};

// 构造函数
Device::Device(const std::string& deviceName)
    : dev(deviceName), nodeId(0), deviceId(0) {}

// 拷贝构造函数
Device::Device(const Device& other)
    : dev(other.dev), nodeId(other.nodeId), deviceId(other.deviceId) {}

// 析构函数
Device::~Device() {}

// 重载 << 运算符
std::ostream& operator<<(std::ostream& os, const Device& obj) {
    os << "Device: " << obj.dev;
    return os;
}

// 获取设备名称
std::string Device::getDeviceName() const {
    return dev;
}

#endif // DEVICE_H
