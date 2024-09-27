// rope_cuda.h

#ifndef ROPE_CUDA_H
#define ROPE_CUDA_H

// 声明 rmsnorm 函数，使其可以被 C++ 程序调用
void rotary_positional_embedding_cuda(int pos, float *vec, int dim, int head_size);

#endif // ROPE_CUDA_H