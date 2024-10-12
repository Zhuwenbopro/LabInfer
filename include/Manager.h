// Manager.h
#ifndef MANAGER_H
#define MANAGER_H

#include <string>
#include <memory>
#include "DeviceManager.h"
#include "Function.h"

class Manager {
private:
    // 私有构造函数
    Manager();

    // 添加 DeviceManager 的引用
    DeviceManager& deviceManager;

public:
    // 静态方法，提供全局访问点
    static Manager& getInstance() {
        static Manager instance;
        return instance;
    }

    // 防止拷贝构造和赋值
    Manager(const Manager&) = delete;
    Manager& operator=(const Manager&) = delete;

    // For Layer
    Function& getFunction(const std::string& deviceName);

    // 注册全局内存，供所有的 layer 使用
    void RegisteMem(const std::shared_ptr<float[]>& ptr);

    std::shared_ptr<float[]> allocate(const size_t size, const std::string& deviceName);

    void toDevice(std::shared_ptr<float[]>& ptr, const size_t size, 
                                    std::string& from_dev, const std::string& to_dev);

    std::shared_ptr<float[]> deepCopy(const std::shared_ptr<float[]>& ptr, 
                                    size_t size, const std::string& deviceName);
};

#endif // ! MANAGER_H
