// rope_cuda.h

#ifndef ROPE_CUDA_H
#define ROPE_CUDA_H

void apply_rope_cuda(float *x, const int *pos, const float *inv_freq, const int n, int dim, const int num);

#endif // ROPE_CUDA_H