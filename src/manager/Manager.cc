#include "Manager.h"

Manager::Manager() : deviceManager(DeviceManager::getInstance()) {
    // 其他初始化代码
}

void Manager::toDevice(Variable& variable, const std::string& deviceName) {

    if(variable.Device() == deviceName) 
        return;

    size_t size = variable.Size();
    // 单机多卡可以直接传数据，还是由本CPU传指令，直接传数据到设备内存中就行
    Device* fromDevice = deviceManager.getDevice(variable.Device());
    Device* toDevice = deviceManager.getDevice(deviceName);
    
    if(toDevice == nullptr)
    // 在这报错
        return;

    float* from_data = variable.Data();
    float* to_data = toDevice->allocate(size);
    // std::cout << from_data << "   " << to_data << std::endl;


    if(variable.Device() != "cpu"){
        if(deviceName == "cpu") { // non-cpu to cpu
            fromDevice->move_out(from_data, to_data, size);
            // std::cout << "cuda -> cpu" <<from_data << "   " << to_data << std::endl;
        }else { // non-cpu to non-cpu
            Device* CPU = deviceManager.getDevice("cpu");
            float* cpu_data = CPU->allocate(size);
            fromDevice->move_out(from_data, cpu_data, size);
            toDevice->move_in(to_data, cpu_data, size);
            CPU->deallocate(cpu_data);
        }
    }else { // cpu to non-cpu
        toDevice->move_in(to_data, from_data, size);
    }
    
    // 执行其他操作
    fromDevice->deallocate(from_data);
    variable.setData(to_data);
    variable.setDevice(deviceName);
}

Function& Manager::getFunction(const std::string& deviceName) {
    Device* dev = deviceManager.getDevice(deviceName);
    return *(dev->F);
}

Device& Manager::getDevice(const std::string& deviceName) {
    return *(deviceManager.getDevice(deviceName));
}