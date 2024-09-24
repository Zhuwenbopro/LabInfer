// matmul.h

#ifndef MATMUL_H
#define MATMUL_H

// 声明 matmul_cuda (xout = Wx) 函数，使其可以被 C++ 程序调用
void matmul_cuda(float *xout, const float *x, const float *w, int n, int d);


#endif // MATMUL_H
