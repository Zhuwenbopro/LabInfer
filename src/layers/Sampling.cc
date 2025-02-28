#include "layers/Sampling.h"

Sampling::Sampling(
    float temperature, 
    int topK, 
    float topP, 
    bool sampling, 
    const std::string& _name
) : t(temperature), k(topK), p(topP), do_sampling(sampling), Layer("cpu", _name) { }

void Sampling::forward(InputWarp& inputWarp) {
    if(do_sampling) {
        F->topK_topP_sampling(inputWarp.output_ids, inputWarp.inter_value, 
            t, k, p, inputWarp.inter_value.ElemLen(), inputWarp.inter_value.ElemNum());
    } else {
        F->max_index(inputWarp.output_ids, inputWarp.inter_value, inputWarp.inter_value.ElemLen(), 1);
    }
}