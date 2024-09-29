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
// 如果有更多的头文件，继续添加
// #include "another_function.h"

class CUDAFunction : public Function {
public:
    void matmul(float *xout, const float *x, const float *w, int n, int d) override {
        matmul_cuda(xout, x, w, n, d);
    }

    void rmsnorm(float* output, const float* input, const float* weight, const float epsilon, int size) override {
        rmsnorm_cuda(output, input, weight, epsilon, size);
    }

    void softmax(float *x, int n) override {
        softmax_cuda(x, n);
    }

    void rotary_positional_embedding(int pos, float *vec, int dim, int head_size) override {
        rotary_positional_embedding_cuda(pos, vec, dim, head_size);
    }

    void silu(float *x, const int n) override {
        silu_cuda(x, n);
    }
};

#endif // CUDA_FUNCTION_H
