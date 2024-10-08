// add_cuda.h

#ifndef ADD_CUDA_H
#define ADD_CUDA_H

// 声明 matmul_cuda (xout = Wx) 函数，使其可以被 C++ 程序调用
void add_cuda(float* y, const float* x1, const float* x2, const int n);


#endif // ADD_CUDA_H
