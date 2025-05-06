#include "Engine.h"

int main() {
    // 获取单例实例
    Engine* engine = Engine::getInstance();
    
    // 初始化引擎
    engine->init();

    // 获取设备管理器实例
    DeviceManager& deviceManager = DeviceManager::getInstance();

    // 使用设备管理器进行内存分配和操作
    std::shared_ptr<void> mem = deviceManager.allocate(1024, "cpu");
    
    // 释放内存
    deviceManager.FreeMem("cpu");

    return 0;
}