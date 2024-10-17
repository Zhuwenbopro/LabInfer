#ifndef TENSOR_H
#define TENSOR_H

#include "Variable.h"


class Tensor : public Variable {
public:
    // 构造函数
    // Tensor 的 _shape 会略去 seq 这个维度，在 decode 时默认为1，prefill 时为 const std::vector<size_t>& _seq 不记录在 _shape 中
    // [batch, seq, ...] 可能会在 seq 出现不同
    Tensor(const std::string& _name, const std::vector<size_t>& _shape, 
        const std::string& _device = "cpu", bool _malloc_mem = false, const std::vector<size_t>& _seq = {1}) 
        : Variable(_name, _shape, _device), seq(_seq), elem_num(0), elem_size(1) {

        if(shape[0] != seq.size()) {
            std::cout << "shape[0] = " << shape[0] << "  vs  seq.size() = " << seq.size() << std::endl;
            throw std::logic_error("tensor shape[0] batch != seq.size()");
        }

        for(int j = 1; j < _shape.size(); j++) {
            elem_size *= _shape[j];
        }

        for(int j = 0; j < _seq.size(); j++) {
            elem_num += _seq[j];
        }

        size = elem_num * elem_size;

        if(_malloc_mem) {
            Manager& manager = Manager::getInstance();
            value = manager.allocate(size, device);
        }
    }

    // 拷贝构造函数（浅拷贝）
    Tensor(const Tensor& other) : Variable(other) {
        // 由于希望进行浅拷贝，仅复制指针和基本信息，不创建新的数据副本
    }

    // 拷贝赋值运算符（浅拷贝）
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            // 调用基类的赋值运算符，进行浅拷贝
            Variable::operator=(other);
        }
        return *this;
    }

    // 深拷贝函数
    Tensor copy() const {
        Tensor copy_tensor = Tensor(name, shape, device, false, seq);
        copy_tensor._copy(*this);
        
        return copy_tensor;
    }

    // 虚析构函数
    ~Tensor() override { }

    const std::vector<size_t>& Seq() const { return seq; }
    const size_t elemNum() const { return elem_num; }
    const size_t elemLen() const { return elem_size; }
private:
    std::vector<size_t> seq;
    size_t elem_size;
    size_t elem_num;
};

#endif