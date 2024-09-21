// rmsnorm.h

#ifndef FUNCRION_CUDA_H
#define FUNCRION_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

// 声明 rmsnorm 函数，使其可以被 C++ 程序调用
void rmsnorm_cuda(float *output, const float *input, const float *weight, int size);

#ifdef __cplusplus
}
#endif

#endif // FUNCRION_CUDA_H
