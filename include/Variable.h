#ifndef VARIABLE_H
#define VARIABLE_H

#include <iostream>
#include <string>
#include <vector>
#include <memory>


/**
    在实现上我应该避免做很多华丽、便捷的接口机制，
    当前主要任务是完成整体流程的运行，因此我仅应该完成最小化的实现。

    1. value、shape、device 初始化
    2. to 设备传输功能，当前仅实现单机多卡之间的设备传输

 */

class Variable {
public:
    // 虚析构函数，确保派生类的析构函数被调用
    virtual ~Variable();


    // 隐式转换 Variable 类和 float*
    operator float*() const { return value.get(); }
    float* rawPtr() const { return value.get(); }
    std::shared_ptr<float[]> sharedPtr() const { return value; }
    void setValue(std::shared_ptr<float[]>& val) { value = val; }
    // void load_data(const std::string& filePath);

    const std::vector<size_t>& Shape() const { return shape; }
    size_t Size() const { return size; }
    const std::string& Device() const { return device; }

    const std::string& Name() const { return name; }
    void setName(const std::string& _name){ name = _name; }

    // 实现设备间传输方法
    void to(const std::string& new_dev);

protected:
    // value 是一个 std::shared_ptr<float[]> 对象，不是一个原始指针。
    // value 持有一个指向 float 类型对象的指针，即 float*。这个指针指向动态分配的内存空间。
    std::shared_ptr<float[]> value;           // 数据指针
    std::vector<size_t> shape;                // 数据形状
    size_t size;                              // 数据大小（元素个数）
    std::string name;                         // 变量名称
    std::string device;                          // 设备

    // 构造函数
    Variable(const std::string& _name, const std::vector<size_t>& _shape, const std::string& _device, bool _malloc_mem);
    
    // 深拷贝
    void _copy(const Variable& from);
};


#endif // VARIABLE_H
