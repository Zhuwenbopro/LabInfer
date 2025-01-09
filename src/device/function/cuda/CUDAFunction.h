// CudaFunctionLibrary.h
#ifndef CUDA_FUNCTION_H
#define CUDA_FUNCTION_H

// 包含所有相关的头文件
#include "Function.h"
#include "matmul_cuda.h"
#include "rmsnorm_cuda.h"
#include "softmax_cuda.h"
#include "rope_cuda.h"
#include "silu_cuda.h"
#include "add_cuda.h"
#include "embedding_cuda.h"
#include "attention_cuda.h"
#include "max_index_cuda.h"
// 如果有更多的头文件，继续添加

void repeat_kv_cuda(float* o, float* in, int dim, int rep, int n);


class CUDAFunction : public Function {
public:
    void whereami() override {
        std::cout << "Function in CUDA" << std::endl;
    }

    void matmul(float *y, const float *x, const float *w, const int n, const int d, const int num) override {
        matmul_cuda(y, x, w, n, d, num);
    }

    void rmsnorm(float* x, const float* w, const int n, int batch_size = 1, const float epsilon=1e-5) override {
        rmsnorm_cuda(x, w, n, batch_size, epsilon);
    }

    void softmax(float *x, const int n, const int batch_size = 1) override {
        softmax_cuda(x, n, batch_size);
    }
    
    void apply_rope(float *x, const int *pos, const float *inv_freq, const int n, int dim, const int num) override {
        apply_rope_cuda(x, pos, inv_freq, n, dim, num);
    }

    void silu(float *x, const int n, const int batch_size = 1) override {
        silu_cuda(x, n, batch_size);
    }
    
    void add(float* y, const float* x1, const float* x2, const int n, const int batch_size = 1) override {
        add_cuda(y, x1, x2, n, batch_size);
    }
    
    void embedding(float* y, const int* x, const float* W, const int d, const int x_size) override {
        embedding_cuda(y, x, W, d, x_size);
    }
    
    void masked_attention(float* y, float* q, float* k, float* v, float* scores, int* pos, int dim, int head_num, int seq_q, int seq_kv) override {
        masked_attention_cuda(y, q, k, v, scores, pos, dim, head_num, seq_q, seq_kv);
    }

    void elem_multiply(float* y, const float* x1, const float* x2, const int size) override {
        elem_multiply_cuda(y, x1, x2, size);
    }
    
    void max_index(float* index, float* x, const int n, const int num) override {
        max_index_cuda(index, x, n, num);
    }

    void repeat_kv(float* o, float* in, int dim, int rep, int n) override {
        repeat_kv_cuda(o, in, dim, rep, n);
    }

};

#endif // CUDA_FUNCTION_H
