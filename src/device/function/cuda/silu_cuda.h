// silu_cuda.h

#ifndef SILU_CUDA_H
#define SILU_CUDA_H

// 声明 rmsnorm 函数，使其可以被 C++ 程序调用
void silu_cuda(float *x, const int n);

#endif // SILU_CUDA_H