// matmul.h

#ifndef MATMUL_CUDA_H
#define MATMUL_CUDA_H

// 声明 matmul_cuda (xout = Wx) 函数，使其可以被 C++ 程序调用
void matmul_cuda(float *xout, const float *x, const float *w, int n, int d, int batch);


#endif // MATMUL_CUDA_H
