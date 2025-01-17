#include "layers/Max.h"

Max::Max(const size_t _vocab_size) : Layer("cpu", "max"), vocab_size(_vocab_size) { }

void Max::forward(InputWarp& inputWarp) {
    Tensor x = inputWarp.inter_value;
    size_t num_offeet = x.ElemNum() - 1;
    // TODO : batch_size > 1 
    F->max_index(inputWarp.output_ids, x + vocab_size*num_offeet, vocab_size, 1);
}