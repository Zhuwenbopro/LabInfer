#include "Device.h"

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
