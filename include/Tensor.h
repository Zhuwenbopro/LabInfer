#ifndef TENSOR_H
#define TENSOR_H

#include "Variable.h"


class Tensor : public Variable {
public:
    // 构造函数
    // Tensor 的 _shape 会略去 seq 这个维度，在 decode 时默认为1，prefill 时为 const std::vector<size_t>& _seq 不记录在 _shape 中
    Tensor(const std::string& _name, float* _value, const std::vector<size_t>& _shape, 
        const std::string& _device = "cpu", bool _malloc_mem = false, const std::vector<size_t>& _seq = {}) 
        : Variable(_name, _value, _shape, _device, _malloc_mem), seq(_seq) {
            if(_seq.size() != 0 && _seq.size() != 1) {
                std::cout << " recalculate tensor size because seqs in batch are not the same size " << std::endl;
                int area = 1;
                for(int j = 1; j < _shape.size(); j++) {
                    area *= _shape[j];
                }
                size = 0;
                for(int j = 0; j < _seq.size(); j++) {
                    size += _seq[j] * area;
                }
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
        Tensor res = Tensor(name, _copy(), shape, device);
        res.to(device);
        // 创建并返回新的 Tensor 对象
        return res;
    }

    // 虚析构函数
    ~Tensor() override { }

    std::vector<size_t> seq;
};

#endif