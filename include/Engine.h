#ifndef ENGINE_H
#define ENGINE_H

#include "DeviceManager.h"

class Engine
{
private:
    // 私有构造函数和析构函数，防止外部实例化和销毁
    Engine();
    ~Engine();

    // 禁用拷贝构造和赋值操作
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // 静态实例指针
    static Engine* instance;

    static DeviceManager& deviceManager;

public:
    // 获取单例实例的静态方法
    static Engine* getInstance();


    // 初始化引擎
    void init();
};


#endif // ENGINE_H