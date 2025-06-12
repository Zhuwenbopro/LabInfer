#include "registry.h"
#include "CUDA/CUDAUtils.h"
#include <iostream>

// cublasHandle_t handle;
// TODO: 让这个函数真正跑起来
void cuda_fp32_linear_exec(void *y, void *x, void *w, int W_in, int W_out, int num)
{
    std::cout << "cuda_fp32_linear_exec" << std::endl;
    // 参数设置
    // float alpha = 1.0f;
    // float beta = 0.0f;

    // CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
    //                          (float *)W_out, num, (float *)W_in,        // M, N, K
    //                          &alpha,
    //                          (float *)w, W_in,                          // A lda
    //                          (float *)x, W_in,                          // B ldb
    //                          &beta,
    //                          (float *)y, (float *)W_out));              // C ldc
}

REGISTER_OP_FUNCTION(Linear, CUDA, FLOAT32, cuda_fp32_linear_exec);