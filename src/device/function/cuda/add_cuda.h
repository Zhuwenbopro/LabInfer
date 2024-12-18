// add_cuda.h

#ifndef ADD_CUDA_H
#define ADD_CUDA_H

// 声明 matmul_cuda (xout = Wx) 函数，使其可以被 C++ 程序调用
void add_cuda(float* y, const float* x1, const float* x2, const int n, const int batch_size);
void add_cuda(float**y, float**x1, float**x2, int n, int num);

#endif // ADD_CUDA_H
