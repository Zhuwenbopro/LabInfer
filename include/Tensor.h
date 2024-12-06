#ifndef TENSOR_H
#define TENSOR_H

#include "Parameter.h"
#include <cassert>
#include <numeric>

class Tensor;

class Tensor : public Variable {
public:
    // 给最开始的人用的, name 不需要，device 等到用到的时候手动.to() 
    Tensor(std::vector<std::vector<size_t>> input_ids, const std::string& dev = "cpu") : Variable(0, 1, dev, "name"), pos("pos", 0, 1, device) {
        elem_num = 0;
        for (const auto& vec : input_ids) {
            elem_num += vec.size();
            seq_len.push_back(vec.size());
        }
        size = elem_num * elem_len;

        value = manager.allocateShared(size, "cpu", "tensor");

        int _index = 0;
        for(int i = 0; i < input_ids.size(); i++) {
            for(int j = 0; j < input_ids[i].size(); j++) {
                value[_index++] = input_ids[i][j];
            }
        }
        (*this).to(dev);
        pos = Parameter("pos", elem_num, 1, device);
    }

    // 给 layer 用的， _malloc_mem 就是 true？
    Tensor(const Tensor& _t, const size_t elemLen) : Variable(_t.elemNum(), elemLen, _t.Device(), "name"), uid(_t.uid), 
                                                seq_len(_t.seq_len), pos(Parameter("pos", _t.elemNum(), 1, _t.Device())) {

        value = manager.allocateShared(size, device, "tensor");
        addPos(_t.pos_arr);
    }

    Tensor(const size_t _eleNum, const size_t _elemLen, const std::string& _device, std::vector<size_t> _uid, std::vector<size_t> _seq_len)
                    : Variable(_eleNum, _elemLen, _device, "name"), uid(_uid), seq_len(_seq_len), pos(Parameter("pos", _eleNum, 1, _device)) {
        value = manager.allocateShared(size, device, "tensor");
    }

    // Tensor& operator=(const Tensor& other) = delete;
    // Tensor& operator=(Tensor&& other) = delete;
    // Tensor(Tensor&& other) = delete;
    // Tensor(Tensor& other) = delete;

    // 深拷贝函数
    void copy(Tensor& t) {
        if(device != t.Device())
            throw std::logic_error("tensor copy device does not match!\n");

        if(size != t.Size())
            throw std::logic_error("tensor copy size does not match!\n");
        
        manager.copy(t, value.get(), size, device);
    }

    // TODO : 想个更好的
    // 不熟悉代码的不要用这个函数
    void copy(const Tensor& from, const size_t length, const size_t from_offset, const size_t to_offset) {
        if(device != from.Device()) throw std::logic_error("copy value not on the same device."); 
        manager.copy(from + from_offset, value.get() + to_offset, length, device);
    }

    // 虚析构函数
    ~Tensor() override {
        value.reset();
    }

    void addPos(const std::vector<std::vector<size_t>>& _pos) {

        pos.setValue(manager.allocateShared(elem_num, "cpu"));
        pos_arr = _pos;

        int _index = 0;
        for(int i = 0; i < _pos.size(); i++) {
            for(int j = 0; j < _pos[i].size(); j++) {
                if(_index >= elem_num) throw std::logic_error("pos oversizes tensor's."); 
                pos[_index++] = _pos[i][j];
            }
        }

        if(_index < elem_num)  throw std::logic_error("pos's size less than tensor's."); 

        pos.to(device);
    }

    const std::vector<size_t>& Uid() const { return uid; }
    const std::vector<size_t>& SeqLen() const { return seq_len; }
    Parameter Pos() { return pos; }
    std::vector<std::vector<size_t>> Position() { return pos_arr; }


    void setUid(const std::vector<size_t>& _uid) { uid = _uid; }

    void setSeqLen(const std::vector<size_t>& _seq_len) { seq_len = _seq_len; }

    void to(const std::string& new_dev) override {
        if (new_dev == device) return;
        if(new_dev == "") throw std::logic_error("there is no device " + new_dev);
        
        manager.toDevice(value, size, device, new_dev);
        device = new_dev;

        if(pos != nullptr) { pos.to(new_dev); }
    }

    Tensor tail() {
        if(seq_len.size() != uid.size())
            throw std::logic_error(name + " tensor.tail() seq_len size does not match uid size"); 
        size_t batch_size = uid.size();
        std::vector<size_t> seq_len_tmp(batch_size, 1);
        Tensor res(batch_size, elem_len, device, uid, seq_len_tmp);

        std::vector<std::vector<size_t>> pos_new;
        for (auto &row : pos_arr) {
            pos_new.push_back({row.back()+1});
        }
        res.addPos(pos_new);

        int len = 0;
        for(int i = 0; i < batch_size; i++) {
            len += seq_len[i] - 1;
            res.copy(*this, elem_len, len*elem_len, i*elem_len);
            len++;
        }

        if(len != elem_num)
            throw std::logic_error(name + " tensor.tail() seq_len sum does not match elem num"); 

        return res;
    }

private:
    std::vector<std::vector<size_t>> pos_arr;
    std::vector<size_t> seq_len;
    std::vector<size_t> uid;
    Parameter pos;
};


#endif