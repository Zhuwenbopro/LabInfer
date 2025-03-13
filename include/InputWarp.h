#ifndef INPUTWARP_H
#define INPUTWARP_H
#include "Tensor.h"

class InputWarp {
private:
    std::string device;
public:
    Tensor<int> pos;
    Tensor<int> input_ids;
    Tensor<int> output_ids;
    Tensor<float> inter_value;
    size_t uid;                 // 会自动构成
    size_t start_pos;

    InputWarp(std::vector<int>& _input_ids, const size_t _uid = 0, const size_t _start_pos = 0) {
        device = "cpu";
        uid = _uid;

        input_ids = Tensor<int>(_input_ids.size(), 1, "cpu", "position");
        pos = Tensor<int>(_input_ids.size(), 1, "cpu", "position");
        for(int i = 0; i < _input_ids.size(); i++) {
            pos[i] = start_pos + i;
            input_ids[i] = _input_ids[i];
        }

        output_ids = Tensor<int>(1, 1, "cpu", "output");
    }

    InputWarp(Tensor<int>& _input_ids, const size_t _uid = 0, const size_t _start_pos = 0) {
        device = _input_ids.Device();
        uid = _uid;
        start_pos = _start_pos;

        input_ids = _input_ids;
        pos = Tensor<int>(input_ids.ElemNum(), 1, "cpu", "position");

        for(int i = 0; i < input_ids.ElemNum(); i++) pos[i] = start_pos + i;
        pos.to(input_ids.Device());
        // TODO: add batch_size
        output_ids = Tensor<int>(1, 1, "cpu", "output");
    }

    void to(const std::string& _device) {
        device = _device;
        if(pos != nullptr) pos.to(device);
        if(input_ids != nullptr) input_ids.to(device);
        if(inter_value != nullptr) inter_value.to(device);
        if(output_ids != nullptr) output_ids.to(device);
    }

    std::string Device() { return device; }
};

#endif