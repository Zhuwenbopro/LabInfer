#ifndef VARIABLE_H
#define VARIABLE_H

#include <iostream>
#include <string>
#include <vector>
#include <typeinfo>
#include "DeviceManager.h"

template <typename T>
class Variable {
public:
    virtual ~Variable() { value.reset(); }

    // Value() 是原始的指针
    operator T*() const { return (T*)value.get(); }
    // T* Value() { return (T*)value.get(); }
    // std::shared_ptr<void> SharedPtr() { return value; }

    size_t Size() const { return elem_num * elem_len; }
    size_t Bytes() const { return elem_num * elem_len * sizeof(T); }
    const std::string& Device() const { return device; }
    const std::string& Name() const { return name; }

    const size_t ElemNum() const { return elem_num; }
    const size_t ElemLen() const { return elem_len; }

    

    void setValue(const std::shared_ptr<T>& _value) { value = _value; }
    void setName(const std::string& _name){ name = _name; }
    void reset() {
        value.reset();
        name = "";
        elem_len = 0;
        elem_num = 0;
    }

    virtual void to(const std::string& new_dev) {
        throw std::logic_error(name + " to(const std::string& new_dev) not implemented. (This is Variable)\n"); 
    }

    Variable(
        const size_t _num, 
        const size_t _len , 
        const std::string& _device, 
        const std::string& _name,
        bool automalloc = true
    ) : device(_device), elem_len(_len), elem_num(_num), value(nullptr), name(_name) {
        if(_num != 0 && _len != 0 && automalloc) {
            value = deviceManager.allocate(Bytes(), device);
        }
    }

    // dst 就是自己呀!!!
    void copy(const size_t dst_offset, const Variable& src, const size_t src_offset, const size_t size) {
        if(device != src.Device()) {
            throw std::logic_error(name + " copy device does not match!\n"); 
        }
        // 不用做类型测试
        deviceManager.copy((void*)((T*)value.get() + dst_offset), (void*)(src + src_offset), size * sizeof(T), device);
    }

    inline long use_count() const {
        if(value == nullptr) {
            throw std::logic_error(name + " value don't have value!\n"); 
        }
        return value.use_count();
    }

protected:
    std::shared_ptr<void> value;
    std::string name;
    std::string device;
    size_t elem_len;
    size_t elem_num;

    static DeviceManager& deviceManager;
};

template <typename T>
DeviceManager& Variable<T>::deviceManager = DeviceManager::getInstance();

#endif // VARIABLE_H
