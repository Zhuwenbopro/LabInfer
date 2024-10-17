// DeviceManager.h
#ifndef DEVICEMANAGER_H
#define DEVICEMANAGER_H

#include <vector>
#include <iostream>
#include <unordered_map>
#include "Device.h"

// 根据 FLAG 包含对应的设备头文件
#include "CPU.h"

#ifdef USE_CUDA
#include "CUDA.h"
#endif

// 可以根据需要添加更多设备

class DeviceManager {
public:
    // 获取 DeviceManager 的单例
    static DeviceManager& getInstance() {
        static DeviceManager instance;
        return instance;
    }

    // 获取存储 Device 实例的静态成员
    std::unordered_map<std::string, Device*>& getDevices() {
        return devices;
    }

    // 静态方法获取 Device 实例
    Device* getDevice(const std::string& deviceName) {
        auto it = devices.find(deviceName);
        if (it != devices.end()) {
            // 已存在，返回共享实例
            return it->second;
        } else {
            // 不存在，创建新实例并存储
            throw std::logic_error("UNKNOWN device " + deviceName);
            return nullptr;
        }
    }

private:
    std::unordered_map<std::string, Device*> devices;

    // 私有构造函数，初始化设备实例
    DeviceManager() {
        // cpu 是一定有的
        devices["cpu"] = new CPU();

#ifdef USE_CUDA
        devices["cuda"] = new CUDA();
#endif
        // 根据需要添加更多设备
    }

    // 禁止拷贝和赋值
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;
};

#endif // DEVICEMANAGER_H
