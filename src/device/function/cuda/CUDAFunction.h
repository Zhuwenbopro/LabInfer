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
// 如果有更多的头文件，继续添加
// #include "another_function.h"

class CUDAFunction : public Function {
public:
    void whereami() override {
        std::cout << "Function in CUDA" << std::endl;
    }

    void matmul(float *y, const float *x, const float *w, const int n, const int d, const int batch_size = 1) override {
        matmul_cuda(y, x, w, n, d, batch_size);
    }

    void rmsnorm(float* x, const float* w, const int n, int batch_size = 1, const float epsilon=1e-5) override {
        rmsnorm_cuda(x, w, n, batch_size, epsilon);
    }

    void softmax(float *x, const int n, const int batch_size = 1) override {
        softmax_cuda(x, n, batch_size);
    }

    void rotary_positional_embedding(int pos, float *vec, int dim, int head_size, const int batch_size = 1) override {
        rotary_positional_embedding_cuda(pos, vec, dim, head_size, batch_size);
    }

    void silu(float *x, const int n, const int batch_size = 1) override {
        silu_cuda(x, n, batch_size);
    }

    void add(float* y, const float* x1, const float* x2, const int n, const int batch_size = 1) override {
        add_cuda(y, x1, x2, n, batch_size);
    }

    void embedding(float* y, const float* x, const float* W, const int d, const int x_size) override {
        embedding_cuda(y, x, W, d, x_size);
    }
};

#endif // CUDA_FUNCTION_H
