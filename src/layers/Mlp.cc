#include "layers/Mlp.h"


Mlp::Mlp(const size_t& _in_size, const size_t& _middle_size) : Layer("cpu", "mlp")
{
    middle_size = _middle_size;
    in_size = _in_size;

    layers.emplace("gate_linear", new Linear(in_size, middle_size, "gate_proj"));
    layers.emplace("up_linear", new Linear(in_size, middle_size, "up_proj"));
    layers.emplace("down_linear", new Linear(middle_size, in_size, "down_proj"));
}


void Mlp::forward(InputWarp& inputWarp)
{
    Tensor x = inputWarp.inter_value;
    layers.at("gate_linear")->forward(inputWarp);
    Tensor gate = inputWarp.inter_value;
    F->silu(gate, gate.Size());
    
    inputWarp.inter_value = x;
    layers.at("up_linear")->forward(inputWarp);
    
    F->elem_multiply(inputWarp.inter_value, inputWarp.inter_value, gate, gate.Size());
    
    layers.at("down_linear")->forward(inputWarp);
}