#include "layers/Embedding.h"


Embedding::Embedding(const size_t _vocab_size, const size_t _hidden_size, const std::string& _name) : Layer("cpu", _name)
{
    vocab_size = _vocab_size;
    hidden_size = _hidden_size;
    
    params.emplace("weight", Parameter<float>(_vocab_size, _hidden_size, "cpu", "weight"));
}

void Embedding::forward(InputWarp& inputWarp)
{
    size_t num = inputWarp.input_ids.ElemNum();
    Tensor<float> y(num, hidden_size, device, name + "_output");
    F->embedding(y, inputWarp.input_ids, params.at("weight"), hidden_size, num);
    inputWarp.inter_value = y;
}
