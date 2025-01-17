#ifndef INPUTWARP_H
#define INPUTWARP_H
#include "Tensor.h"

class InputWarp {
public:
    Tensor<int> pos;
    Tensor<int> input_ids;
    Tensor<int> output_ids;
    Tensor<float> inter_value;
    size_t uid;
    size_t start_pos;

    InputWarp(Tensor<int>& _input_ids, const size_t _start_pos = 0) {
        static size_t guid = 0;
        uid = guid++;
        start_pos = _start_pos;

        input_ids = _input_ids;
        pos = Tensor<int>(input_ids.ElemNum(), 1, "cpu", "position");

        for(int i = 0; i < input_ids.ElemNum(); i++) pos[i] = start_pos + i;
        pos.to(input_ids.Device());
    }

    void to(const std::string& device) {
        if(pos != nullptr) pos.to(device);
        if(input_ids != nullptr) input_ids.to(device);
        if(output_ids != nullptr) output_ids.to(device);
        if(inter_value != nullptr) inter_value.to(device);
    }
};

#endif