#include "Manager.h"

Manager::Manager() : deviceManager(DeviceManager::getInstance()) {
    // 其他初始化代码
}

void Manager::toDevice(std::shared_ptr<float[]>& ptr, const size_t size, 
                                    std::string& from_dev, const std::string& to_dev) {

    // 单机多卡可以直接传数据，还是由本CPU传指令，直接传数据到设备内存中就行
    Device* fromDevice = deviceManager.getDevice(from_dev);
    Device* toDevice = deviceManager.getDevice(to_dev);
    
    std::shared_ptr<float[]> to_value = allocate(size, to_dev);

    if(from_dev != "cpu"){
        if(to_dev == "cpu") { // non-cpu to cpu
            fromDevice->move_out(ptr.get(), to_value.get(), size);
        }else { // non-cpu to non-cpu
            Device* CPU = deviceManager.getDevice("cpu");
            float* cpu_data = CPU->allocate(size);
            fromDevice->move_out(ptr.get(), cpu_data, size);
            toDevice->move_in(to_value.get(), cpu_data, size);
            CPU->deallocate(cpu_data);
        }
    }else { // cpu to non-cpu
        toDevice->move_in(to_value.get(), ptr.get(), size);
    }
    
    ptr = to_value;
}

Function& Manager::getFunction(const std::string& deviceName) {
    Device* dev = deviceManager.getDevice(deviceName);
    return *(dev->F);
}

std::shared_ptr<float[]> Manager::allocate(const size_t size, const std::string& deviceName) {
    Device * dev = deviceManager.getDevice(deviceName);
    float* raw_ptr = dev->allocate(size);

    // 使用 Lambda 捕获 dev，并作为删除器
    auto deleter = [dev](float* ptr) {
        if (ptr) {
            // dev->whoami();
            std::cout << "deleter: 释放 float 数组内存\n";
            dev->deallocate(ptr);
        }
    };

    return std::shared_ptr<float[]>(raw_ptr, deleter);
}

// 深度复制函数
std::shared_ptr<float[]> Manager::deepCopy(const std::shared_ptr<float[]>& ptr, size_t size, const std::string& deviceName) {
    if (!ptr) {
        return nullptr;
    }

    std::shared_ptr<float[]> cptr = allocate(size, deviceName);

    Device * dev = deviceManager.getDevice(deviceName);
    dev->copy(ptr.get(), cptr.get(), size);

    return cptr;
}