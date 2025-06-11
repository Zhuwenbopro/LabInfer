#pragma once
#include "Function.h"
#include "CUDAUtils.h"
#include <iostream>


class CUDAFunction : public Function {
public:

    CUDAFunction();
    ~CUDAFunction();
    void whereami() override {
        std::cout << "Function in CUDA" << std::endl;
    }

    void matmul(float *y, const float *x, const float *w, const int n, const int d, const int num) override;
    void rmsnorm(float* x, const float* w, const int n, int batch_size = 1, const float epsilon=1e-5) override;
    void softmax(float *x, const int n, const int batch_size = 1) override;
    void apply_rope(float *x, const int *pos, const float *inv_freq, const int n, int dim, const int num) override;
    void silu(float *x, const int n, const int batch_size = 1) override;
    void add(float* y, const float* x1, const float* x2, const int n, const int batch_size = 1) override;
    void embedding(float* y, int* x, const float* W, const int d, const int x_size) override;
    void masked_attention(float* y, float* q, float* k, float* v, float* scores, int* pos, int dim, int head_num, int seq_q, int seq_kv) override;
    void elem_multiply(float* y, const float* x1, const float* x2, const int size) override;
    void max_index(int* index, float* x, const int n, const int num) override;
    void repeat_kv(float* o, float* in, int dim, int rep, int n) override;
    void topK_topP_sampling(int* index, float* logits, float temperature, int topK, float topP, int n, int num) override;
};