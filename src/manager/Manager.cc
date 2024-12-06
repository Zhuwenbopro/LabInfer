#include "Manager.h"

DeviceManager& Manager::deviceManager = DeviceManager::getInstance();

void Manager::toDevice(std::shared_ptr<float[]>& ptr, const size_t size, 
                                    std::string& from_dev, const std::string& to_dev, const std::string& name) {
    if(from_dev == to_dev) return;
    // 单机多卡可以直接传数据，还是由本CPU传指令，直接传数据到设备内存中就行
    Device* fromDevice = deviceManager.getDevice(from_dev);
    Device* toDevice = deviceManager.getDevice(to_dev);
    
    std::shared_ptr<float[]> to_value = allocateShared(size, to_dev, name);

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

std::shared_ptr<float[]> Manager::allocateShared(const size_t size, const std::string& deviceName, const std::string& name) {
    Device * dev = deviceManager.getDevice(deviceName);
    float* raw_ptr = dev->allocate(size);

    // 使用 Lambda 捕获 dev，并作为删除器
    auto deleter = [dev, name](float* ptr) {
        if (ptr) {
            // std::cout << name << " deallocate" << std::endl;
            // dev->whoami();
            dev->deallocate(ptr);
        }
    };

    return std::shared_ptr<float[]>(raw_ptr, deleter);
}

float* Manager::allocateRaw(const size_t size, const std::string& device) {
    Device * dev = deviceManager.getDevice(device);
    return dev->allocate(size);
}

std::shared_ptr<float[]> Manager::deepCopy(const std::shared_ptr<float[]>& ptr, size_t size, const std::string& device) {
    if (!ptr) {
        throw std::logic_error("ptr in Manager::deepCopy() is null");
        return nullptr;
    }
    std::shared_ptr<float[]> cptr = allocateShared(size, device);
    Device * dev = deviceManager.getDevice(device);
    dev->copy(ptr.get(), cptr.get(), size);

    return cptr;
}


void Manager::copy(float* from, float* to, size_t size, std::string& device) {
    if (!from || !to) {
        throw std::logic_error("ptr in Manager::deepCopy() is null");
        return ;
    }
    Device * dev = deviceManager.getDevice(device);
    dev->copy(from, to, size);
}

void Manager::RegisteMem(const std::string& name, const std::shared_ptr<float[]>& ptr) {
    shared_mem[name] = ptr;
}

std::shared_ptr<float[]>& Manager::GetMem(const std::string& name) {
    return shared_mem.at(name);
}

bool Manager::FindMem(const std::string& name) {
    auto it = shared_mem.find(name);
    return (it != shared_mem.end());
}