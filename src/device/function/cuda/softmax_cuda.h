// softmax_cuda.h

#ifndef SOFTMAX_CUDA_H
#define SOFTMAX_CUDA_H

// 声明 rmsnorm 函数，使其可以被 C++ 程序调用
void softmax_cuda(float *x, int n, int batch_size);
void softmax_cuda(float**x, int n, int num);

#endif // SOFTMAX_CUDA_H
