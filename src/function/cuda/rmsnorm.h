// rmsnorm.h

#ifndef RMSNORM_H
#define RMSNORM_H


#ifdef __cplusplus
extern "C" {
#endif

// 声明 rmsnorm 函数，使其可以被 C++ 程序调用
void rmsnorm_cuda(float *output, const float *input, const float *weight, const float epsilon, int size);

#ifdef __cplusplus
}
#endif

#endif // RMSNORM_H
