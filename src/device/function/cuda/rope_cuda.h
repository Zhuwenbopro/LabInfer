// rope_cuda.h

#ifndef ROPE_CUDA_H
#define ROPE_CUDA_H

void apply_rope_cuda(float *x, const float *pos, const float *cos, const float *sin, const int n, const int dim, const int num);

#endif // ROPE_CUDA_H