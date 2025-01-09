// softmax_cuda.h

#ifndef SOFTMAX_CUDA_H
#define SOFTMAX_CUDA_H

// 声明 rmsnorm 函数，使其可以被 C++ 程序调用
__global__ void softmax_gpu(float *__restrict__ x, int size);
void softmax_cuda(float *x, int n, int batch_size);

#endif // SOFTMAX_CUDA_H
