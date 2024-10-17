// attention_cuda.h

#ifndef ATTENTION_CUDA_H
#define ATTENTION_CUDA_H

void maksed_attention_cuda(float* y, const float* q, const float* k, const float* v, const int dim, const int q_head, const int kv_head, const int _pos);

#endif // ATTENTION_CUDA_H
