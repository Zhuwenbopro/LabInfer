#include "Manager.h"

Manager::Manager() : deviceManager(DeviceManager::getInstance()) {
    // 其他初始化代码
}

void Manager::toDevice(Variable& variable, const std::string& deviceName) {

    if(variable.Device() == deviceName) 
        return;

    // 单机多卡可以直接传数据，还是由本CPU传指令，直接传数据到设备内存中就行
    Device* dev = deviceManager.getDevice(deviceName);
    // 执行其他操作
}

Function& Manager::getFunction(const std::string& deviceName) {
    Device* dev = deviceManager.getDevice(deviceName);
    return *(dev->F);
}

Device& Manager::getDevice(const std::string& deviceName) {
    return *(deviceManager.getDevice(deviceName));
}