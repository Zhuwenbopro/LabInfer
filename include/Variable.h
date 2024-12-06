#ifndef VARIABLE_H
#define VARIABLE_H

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "Manager.h"


/**
    1. value、shape、device 初始化
    2. to 设备传输功能，当前仅实现单机多卡之间的设备传输
 */

class Variable {
public:
    // 虚析构函数，确保派生类的析构函数被调用
    virtual ~Variable() { }

    // 隐式转换 Variable 类和 float*
    operator float*() const { return value.get(); }
    float* rawPtr() const { return value.get(); }
    std::shared_ptr<float[]> sharedPtr() const { return value; }

    void setValue(const std::shared_ptr<float[]>& val) { value = val; }
    void setName(const std::string& _name){ name = _name; }

    virtual void to(const std::string& new_dev) {
        throw std::logic_error(name + " to(const std::string& new_dev) not implemented."); 
    }
    
    size_t Size() const { return size; }
    const std::string& Device() const { return device; }
    const std::string& Name() const { return name; }

    const size_t elemNum() const { return elem_num; }
    const size_t elemLen() const { return elem_len; }

protected:
    // value 是一个 std::shared_ptr<float[]> 对象，不是一个原始指针。
    // value 持有一个指向 float 类型对象的指针，即 float*。这个指针指向动态分配的内存空间。
    std::shared_ptr<float[]> value;           // 数据指针
    std::string name;                         // 变量名称
    std::string device;                          // 设备

    size_t size;
    size_t elem_len;
    size_t elem_num;

    static Manager& manager;

    // TODO : 将size与elem_len、elem_num解耦，可预先分配
    // 构造函数
    Variable(const size_t _num, const size_t _len , const std::string& _device, const std::string& _name) : size(_num * _len), 
                                                device(_device), elem_len(_len), elem_num(_num), value(nullptr), name(_name) { }
    
    // 深拷贝
    // FIXME : to be delete
    void _copy(const Variable& from) {
        value = manager.deepCopy(from.sharedPtr(), size, device);
    }
};

#endif // VARIABLE_H
