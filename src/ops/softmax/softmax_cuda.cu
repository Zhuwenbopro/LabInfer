#include "registry.h"
#include "CUDA/CUDAUtils.h"

__global__ void softmax_gpu(float *__restrict__ x, int size) {
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    int batch_idx = blockIdx.y;
    int idx = batch_idx * size;

    x += idx;

    // 找到最大值（用于数值稳定性）
    float max_val = -FLT_MAX;
    for (int i = tid; i < size; i += block_size) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float shared_max;

    float max_result = BlockReduce(temp_storage).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0) {
        shared_max = max_result;
    }
    __syncthreads();
    max_val = shared_max;

    // 计算指数和总和
    float sum = 0.0f;
    for (int i = tid; i < size; i += block_size) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    sum = BlockReduce(temp_storage).Sum(sum);
    if (threadIdx.x == 0) {
        shared_max = sum;
    }
    __syncthreads();
    sum = shared_max;

    // 归一化
    for (int i = tid; i < size; i += block_size) {
        x[i] /= sum;
    }
}


void cuda_fp32_softmax_exec(void *x, int n, int num)
{
    dim3 blockDim(num_threads_large);
    dim3 gridDim(1, num);
    
    softmax_gpu<<<gridDim, blockDim>>>((float*)x, n);
}

REGISTER_OP_FUNCTION(Softmax, CUDA, FLOAT32, cuda_fp32_softmax_exec);