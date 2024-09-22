// matmul.h

#ifndef MATMUL_H
#define MATMUL_H


#ifdef __cplusplus
extern "C" {
#endif

// 声明 matmul_cuda (xout = Wx) 函数，使其可以被 C++ 程序调用
void matmul_cuda(float *xout, const float *x, const float *w, int n, int d);


#ifdef __cplusplus
}
#endif

#endif // MATMUL_H
