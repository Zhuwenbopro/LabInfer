#ifndef TENSOR_H
#define TENSOR_H

#include "Variable.h"
#include <cassert>
#include <numeric>


class Tensor : public Variable {
public:
    // 给最开始的人用的, name 不需要，device 等到用到的时候手动.to() 
    Tensor(std::vector<std::vector<size_t>> input_ids) : Variable(0, 1, "cpu") {

        batch_size = input_ids.size();

        static size_t guid = 0;
        elem_num = 0;
        for (const auto& vec : input_ids) {
            elem_num += vec.size();
            seq.push_back(vec.size());
            uid.push_back(++guid);
        }
        size = elem_num * elem_len;

        Manager& manager = Manager::getInstance();
        value = manager.allocateShared(size, device);

        int _index = 0;
        for(int i = 0; i < input_ids.size(); i++) {
            for(int j = 0; j < input_ids[i].size(); j++) {
                value[_index++] = input_ids[i][j];
            }
        }
    }

    // 给 layer 用的， _malloc_mem 就是 true？
    Tensor(const Tensor& _t, const size_t elemLen) : Variable(_t.elemNum(), elemLen, _t.Device()), 
                                            batch_size(_t.batchSize()), uid(_t.Uid()), seq(_t.Seq()) {
        Manager& manager = Manager::getInstance();
        value = manager.allocateShared(size, device);
    }

    Tensor(const size_t _eleNum, const size_t _elemLen, const std::string& _device,  std::vector<size_t> _uid)
                                     : Variable(_eleNum, _elemLen, _device), batch_size(_uid.size()), uid(_uid) {
        Manager& manager = Manager::getInstance();
        value = manager.allocateShared(size, device);
    }

    // 深拷贝函数
    Tensor copy() const {
        Tensor copy_tensor(*this, elem_len);
        copy_tensor._copy(*this);
        
        return copy_tensor;
    }

    // 虚析构函数
    ~Tensor() override { }

    const std::vector<size_t>& Uid() const { return uid; }
    const std::vector<size_t>& Seq() const { return seq; }  // to be deleted
    const size_t batchSize() const { return batch_size; }         // to be deleted

private:
    size_t batch_size;
    std::vector<size_t> seq;
    std::vector<size_t> uid;
};

#endif