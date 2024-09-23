#ifndef DEVICEFACTORY_H
#define DEVICEFACTORY_H

#include "Device.h"
#include <unordered_map>
#include <iostream>

class DeviceFactory {
public:
    // 静态方法获取 Device 实例
    static Device* getDevice(const std::string& deviceName) {
        auto& devices = getDevices();
        auto it = devices.find(deviceName);
        if (it != devices.end()) {
            // 已存在，返回共享实例
            return it->second;
        } else {
            // 不存在，创建新实例并存储
            auto device = new Device(deviceName);
            devices[deviceName] = device;
            return device;
        }
    }

private:
    // 私有构造函数确保单例
    DeviceFactory() = default;
    DeviceFactory(const DeviceFactory&) = delete;
    DeviceFactory& operator=(const DeviceFactory&) = delete;

    // 获取存储 Device 实例的静态成员
    static std::unordered_map<std::string, Device*>& getDevices() {
        static std::unordered_map<std::string, Device*> devices;
        return devices;
    }

};

#endif // DEVICEFACTORY_H
