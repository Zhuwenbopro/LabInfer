// Manager.h
#ifndef MANAGER_H
#define MANAGER_H

#include <string>
#include <memory>
#include <unordered_map>
#include "DeviceManager.h"
#include "Function.h"

class Manager {
private:
    // 私有构造函数
    Manager();
    // 添加 DeviceManager 的引用
    DeviceManager& deviceManager;
    std::unordered_map<std::string, std::shared_ptr<float[]>> shared_mem;

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
    Function& getFunction(const std::string& device);

    // 注册全局内存，供所有的 layer 使用
    void RegisteMem(const std::string& name, const std::shared_ptr<float[]>& ptr);

    std::shared_ptr<float[]>& GetMem(const std::string& name);

    bool FindMem(const std::string& name);

    // 分配 device 里大小为 size 的内存
    std::shared_ptr<float[]> allocateShared(const size_t size, const std::string& device);

    float* allocateRaw(const size_t size, const std::string& device);

    void toDevice(std::shared_ptr<float[]>& ptr, const size_t size, 
                                    std::string& from_dev, const std::string& to_dev);

    // 深度复制同类型的数据 device 是 ptr 的
    std::shared_ptr<float[]> deepCopy(const std::shared_ptr<float[]>& ptr, 
                                    size_t size, const std::string& device);
};

#endif // ! MANAGER_H
