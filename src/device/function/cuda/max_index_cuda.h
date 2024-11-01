#ifndef MAX_INDEX_CUDA_H
#define MAX_INDEX_CUDA_H

// 声明 rmsnorm 函数，使其可以被 C++ 程序调用
void max_index_cuda(float* index, float *x, int n, int num);

#endif // MAX_INDEX_CUDA_H
