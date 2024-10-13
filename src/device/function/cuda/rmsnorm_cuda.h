// rmsnorm_cuda.h

#ifndef RMSNORM_CUDA_H
#define RMSNORM_CUDA_H

// 声明 rmsnorm 函数，使其可以被 C++ 程序调用
void rmsnorm_cuda(float* x, const float* w, int n, int batch_size, const float epsilon);

#endif // RMSNORM_CUDA_H
