// DeviceManager.h
#ifndef DEVICEMANAGER_H
#define DEVICEMANAGER_H

#include <vector>
#include <iostream>
#include <unordered_map>
#include <memory>
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

    ~DeviceManager() {
        for (auto& pair : shared_mem) {
            pair.second.reset();  // 显式清空 shared_ptr，释放它管理的内存
        }

        for (auto& pair : devices) {
            delete pair.second;
        }

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

    // 分配内存
    std::shared_ptr<void> allocate(size_t bytes, const std::string& deviceName) {
        Device * dev = getDevice(deviceName);
        void* raw_ptr = dev->allocate(bytes);

        auto deleter = [dev](void* ptr) {
            if (ptr) {
                // std::cout << "deallocate " << ptr << std::endl;
                dev->deallocate(ptr);
            }
        };

        // std::cout << "malloc " << _device << "  " << raw_ptr << "   bytes=" << bytes << std::endl;
        return std::shared_ptr<void>(raw_ptr, deleter);
    }

    void copy(void* dst, void* src, size_t bytes, const std::string& deviceName) {
        Device * dev = getDevice(deviceName);
        dev->copy(dst, src, bytes);
    }

    // 转移
    void toDevice(
        std::shared_ptr<void>& value, 
        size_t bytes, 
        const std::string& srcDeviceName, 
        const std::string& dstDeciceName
    ) {
        if(srcDeviceName == dstDeciceName) return;

        std::shared_ptr<void> to_value = allocate(bytes, dstDeciceName);

        if(srcDeviceName != "cpu"){
            Device* fromDevice = getDevice(srcDeviceName);
            if(dstDeciceName == "cpu") { // non-cpu to cpu
                fromDevice->move_out(value.get(), to_value.get(), bytes);
            }else { // non-cpu to non-cpu
                Device* toDevice = getDevice(dstDeciceName);
                std::shared_ptr<void> cpu_tmp = allocate(bytes, "cpu");
                fromDevice->move_out(value.get(), cpu_tmp.get(), bytes);
                toDevice->move_in(to_value.get(), cpu_tmp.get(), bytes);
            }
        }else { // cpu to non-cpu
            Device* toDevice = getDevice(dstDeciceName);
            toDevice->move_in(to_value.get(), value.get(), bytes);
        }

        value = to_value;
    }

    // 注册全局内存，供所有的 layer 使用
    void RegisteMem(const std::string& name, const std::shared_ptr<void>& ptr) {
        shared_mem[name] = ptr;
    }

    std::shared_ptr<void>& GetMem(const std::string& name) {
        return shared_mem.at(name);
    }

    bool FindMem(const std::string& name) {
        auto it = shared_mem.find(name);
        return (it != shared_mem.end());
    }

    void FreeMem(const std::string& name) {
        shared_mem[name].reset();
        shared_mem.erase(name);
    }

private:
    std::unordered_map<std::string, Device*> devices;
    std::unordered_map<std::string, std::shared_ptr<void>> shared_mem;

    // 私有构造函数，初始化设备实例
    DeviceManager() {
        // cpu 是一定有的
        devices["cpu"] = new CPU();
    
    // TODO : add 'cuda:0', 'cuda:1'
#ifdef USE_CUDA
        devices["cuda"] = new CUDA();
#endif
        // TODO : 根据需要添加更多设备
    }

    // 禁止拷贝和赋值
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;

    // 获取存储 Device 实例的静态成员
    std::unordered_map<std::string, Device*>& getDevices() {
        return devices;
    }

};

#endif // DEVICEMANAGER_H
