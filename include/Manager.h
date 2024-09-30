// Manager.h
#ifndef MANAGER_H
#define MANAGER_H

#include <string>
#include "DeviceManager.h"
#include "Variable.h"
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

    void toDevice(Variable& variable, const std::string& deviceName);

    Function& getFunction(const std::string& deviceName);

    // 我还没想好要不要对外开放
    Device& getDevice(const std::string& deviceName);
};

#endif // ! MANAGER_H
