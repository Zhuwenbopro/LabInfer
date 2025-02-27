#include "layers/Linear.h"

Linear::Linear(const size_t size_in, const size_t size_out, const std::string& _name, bool _bias) : Layer("cpu", _name)
{
    input_size = size_in;
    output_size = size_out;
    bias = _bias;
    
    params.emplace("weight", Parameter<float>(size_out, size_in, "cpu", "weight"));

    if(bias) params.emplace("bias", Parameter<float>(1, size_in, "cpu", "bias"));
}

void Linear::forward(InputWarp& inputWarp)
{   
    if(inputWarp.inter_value.ElemLen() != input_size) {
        std::cout << "input size = " << input_size << "   = " << inputWarp.inter_value.ElemLen() << std::endl;
        throw std::runtime_error("Layer " + name + "'s input len not match param len.");
    }
    
    if(name == "lm_head") {
        Tensor<float> y(1, output_size, device, name + "_output");
        int num = inputWarp.inter_value.ElemNum();
        F->matmul(y, inputWarp.inter_value + (num-1)*input_size, params.at("weight"), input_size, output_size, 1);
        inputWarp.inter_value = y;
    } else {
        Tensor<float> y(inputWarp.inter_value.ElemNum(), output_size, device, name + "_output");
        F->matmul(y, inputWarp.inter_value, params.at("weight"), input_size, output_size, inputWarp.inter_value.ElemNum());
        inputWarp.inter_value = y;
    }
}