// rmsnorm.h

#ifndef RMSNORM_H
#define RMSNORM_H

// 声明 rmsnorm 函数，使其可以被 C++ 程序调用
void rmsnorm_cuda(float *output, const float *input, const float *weight, const float epsilon, int size);

#endif // RMSNORM_H
