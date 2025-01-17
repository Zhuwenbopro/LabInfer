#include "layers/RMSNorm.h"

RMSNorm::RMSNorm(const size_t _hidden_size, const float _epsilon, const std::string& _name) : Layer("cpu", _name), hidden_size(_hidden_size), epsilon(_epsilon)
{
    params.emplace("weight", Parameter<float>(1, hidden_size, "cpu", "weight"));
}

void RMSNorm::forward(InputWarp& inputWarp)
{
    F->rmsnorm(inputWarp.inter_value, params.at("weight"), hidden_size, inputWarp.inter_value.ElemNum(), epsilon);
}