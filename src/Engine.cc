#include "Engine.h"

// 初始化静态成员变量
Engine* Engine::instance = nullptr;

Engine::Engine()
{
    // 构造函数逻辑（如果有需要）
}

Engine::~Engine()
{
    // 析构函数逻辑（如果有需要）
}

Engine* Engine::getInstance()
{
    if (instance == nullptr)
    {
        instance = new Engine();
    }
    return instance;
}

DeviceManager& Engine::deviceManager = DeviceManager::getInstance();

void Engine::init()
{
    // 创建模型的运行结构 workers

    // 初始化 worker （装上device、layer初始化，装参数）
}