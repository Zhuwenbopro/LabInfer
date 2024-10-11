// embedding_cuda.h

#ifndef EMBEDDING_CUDA_H
#define EMBEDDING_CUDA_H

void embedding_cuda(float* y, const float* x, const float* W, const int d, const int x_size);

#endif // EMBEDDING_CUDA_H