// attention_cuda.h

#ifndef ATTENTION_CUDA_H
#define ATTENTION_CUDA_H

// void maksed_attention_cuda(float* y, const float* q, const float* k, const float* v, const int dim, const int q_head, const int kv_head, const int _pos);
void masked_attention_cuda(
    float* y, 
    float* q, 
    float* k, 
    float* v, 
    float* scores, 
    int* pos, 
    int dim, 
    int head_num,
    int seq_q,
    int seq_kv
);

#endif // ATTENTION_CUDA_H
