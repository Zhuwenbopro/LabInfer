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

    InputWarp() = delete;
    InputWarp(Tensor<int>& _input_ids) {
        static size_t guid = 0;
        uid = guid++;
        input_ids = _input_ids;
        pos = Tensor<int>(input_ids.ElemNum(), 1, input_ids.Device(), "position");

        for(int i = 0; i < input_ids.ElemNum(); i++) pos[i] = i;
    }

    void to(const std::string& device) {
        pos.to(device);
        input_ids.to(device);
        output_ids.to(device);
        inter_value.to(device);
    }
};

#endif