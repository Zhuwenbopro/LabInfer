// flash_attention_example.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cassert>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>

/**
 * CUDA 的简单错误检查宏
 */
#define CUDA_CHECK(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA Error %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)



/**
 * 该 kernel 会并行覆盖所有分块 (i, j)，其中:
 *  gridDim.x = Tc, gridDim.y = Tr
 *  blockDim.x * blockDim.y = 每块处理 Br×Bc 内部的并行
 *
 *  改动要点：让 S_ij 原地存放 exponent，省去 P_ij 数组。
 */
__global__
void flashAttentionFwdKernelInplace(
    const float* __restrict__ Q,  // [N, d]
    const float* __restrict__ K,  // [N, d]
    const float* __restrict__ V,  // [N, d]
    float* __restrict__ O,        // [N, d]
    float* __restrict__ m,        // [N], row-wise max
    float* __restrict__ l,        // [N], row-wise sum(exp(...))
    int N, int d,
    int Br, int Bc
) {
    // blockIdx -> (i, j) tile
    int j = blockIdx.x; 
    int i = blockIdx.y; 
    int rowStart = i * Br; 
    int colStart = j * Bc;
    if (rowStart >= N || colStart >= N) return;

    int actualBr = min(Br, N - rowStart);
    int actualBc = min(Bc, N - colStart);

    // 指针计算
    const float* Q_i = Q + (rowStart * d);
    const float* K_j = K + (colStart * d);
    const float* V_j = V + (colStart * d);
          float* O_i = O + (rowStart * d);
          float* m_i = m + rowStart;
          float* l_i = l + rowStart;

    extern __shared__ float smem[];
    // 布局:
    // [0..Bc*d-1]          -> K_block
    // [Bc*d..2*Bc*d-1]     -> V_block
    // [2*Bc*d..2*Bc*d + Br*Bc -1] -> S_ij
    // [.. + Br .. + 2*Br-1]       -> rowFactor1
    // [.. + 2*Br .. + 3*Br-1]     -> rowFactor2
    size_t offset = 0;
    float* K_block = smem + offset;
    offset += (Bc * d);

    float* V_block = smem + offset;
    offset += (Bc * d);

    float* S_ij = smem + offset;
    offset += (Br * Bc);

    float* rowFactor1 = smem + offset;
    offset += Br;

    float* rowFactor2 = smem + offset;
    offset += Br;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int stride = blockDim.x * blockDim.y;


    // ---------------------------
    // 1) 加载 K_j, V_j 到 shared memory
    for (int idx = tid; idx < actualBc * d; idx += stride) {
        K_block[idx] = K_j[idx];
        V_block[idx] = V_j[idx];
    }
    __syncthreads();

    // ---------------------------
    // 2) 计算 S_ij = Q_i * K_j^T   (大小: [Br, Bc])
    for (int idx = tid; idx < actualBr * actualBc; idx += stride) {
        int row = idx / actualBc;
        int col = idx % actualBc;
        float sum_val = 0.f;
        for (int kk = 0; kk < d; ++kk) {
            float q_val = Q_i[row * d + kk];
            float k_val = K_block[col * d + kk];
            sum_val += q_val * k_val;
        }
        S_ij[row * Bc + col] = sum_val;
    }
    __syncthreads();

    // ---------------------------
    // 3) row-wise softmax(分块), 并计算 factor1,factor2
    for (int idx = tid; idx < actualBr; idx += stride) {
        float m_old = m_i[idx];
        float l_old = l_i[idx];

        // 3a) 找到该行在 [col=0..actualBc-1] 范围内的最大值
        float row_max_val = -1e30f;
        for(int col = 0; col < actualBc; ++col) {
            float val = S_ij[idx * Bc + col];
            if (val > row_max_val) row_max_val = val;
        }
        float m_new = (row_max_val > m_old) ? row_max_val : m_old;

        // 3b) 做原地 exponent 并计算 sum_exp
        float sum_exp = 0.f;
        for(int col = 0; col < actualBc; ++col) {
            float val = S_ij[idx * Bc + col];
            // in-place: 覆盖S_ij
            val = __expf(val - m_new);
            S_ij[idx * Bc + col] = val;
            sum_exp += val;
        }

        // 3c) 更新 l_new
        float factor = __expf(m_old - m_new);
        float l_new = l_old * factor + sum_exp;

        // 写回
        m_i[idx] = m_new;
        l_i[idx] = l_new;

        // 3d) 计算 factor1, factor2
        float f1 = 0.f;
        if (l_new > 1e-30f) {
            f1 = (l_old * factor) / l_new;
        }
        float f2 = (l_new > 1e-30f) ? (1.f / l_new) : 0.f;

        rowFactor1[idx] = f1;
        rowFactor2[idx] = f2;
    }
    __syncthreads();

    // ---------------------------
    // 4) 更新 O_i： new_o = factor1 * old_o + factor2 * sum_{col}(S_ij[row,col] * V_block[col])
    for (int idx = tid; idx < actualBr * d; idx += stride) {
        int row = idx / d;
        int dim = idx % d;
        float factor1 = rowFactor1[row];
        float factor2 = rowFactor2[row];

        float old_o = O_i[row * d + dim];

        // dot_val = sum_{col in [0..actualBc)}( S_ij[row, col]* V_block[col, dim] )
        float dot_val = 0.f;
        for(int col = 0; col < actualBc; ++col) {
            float p_ij = S_ij[row * Bc + col]; // now holds exponent
            float v_val = V_block[col * d + dim];
            dot_val += (p_ij * v_val);
        }

        float new_o = factor1 * old_o + factor2 * dot_val;
        O_i[row * d + dim] = new_o;
    }
}

//
// In-place版本对应的 fwd 函数
//
void flashAttentionFwd_Inplace(
    const float* Q, // [N, d]
    const float* K, // [N, d]
    const float* V, // [N, d]
    float* O,       // [N, d]
    float* m,       // [N]
    float* l,       // [N]
    int N, int d,
    int Br, int Bc
) {
    // 计算分块数量
    int Tr = (N + Br - 1) / Br; 
    int Tc = (N + Bc - 1) / Bc;

    // block、grid配置
    dim3 block(16, 8);
    dim3 grid(Tc, Tr);

    // 需要的共享内存(去掉了 P_ij 的那块)
    // K_block(Bc*d) + V_block(Bc*d) + S_ij(Br*Bc) + rowFactor1(Br) + rowFactor2(Br)
    size_t smem_floats = (Bc*d) + (Bc*d) + (Br*Bc) + Br + Br;
    size_t smem_size = smem_floats * sizeof(float);

    // 发 kernel
    flashAttentionFwdKernelInplace<<<grid, block, smem_size>>>(
        Q, K, V,
        O, m, l,
        N, d,
        Br, Bc
    );
    CUDA_CHECK(cudaGetLastError());
}


int main() {
    printf("begin...\n");
    // 序列长度 N=2048, 单头的维度 d=64
    const int N = 1024;
    const int d = 2048;

    // 1) 查询可用的 GPU 数量，并指定要使用的设备 ID
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices <= 0) {
        fprintf(stderr, "No CUDA-capable device found.\n");
        return 1;
    }
    int deviceId = 0; // 假设我们用第 0 张 GPU
    CUDA_CHECK(cudaSetDevice(deviceId));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));

    size_t sharedMemPerBlock = prop.sharedMemPerBlock;  
    size_t M_floats = sharedMemPerBlock / sizeof(float);

    int Bc = static_cast<int>(std::ceil(M_floats / (4.0 * d)));
    Bc = std::max(1, Bc);  // 确保至少为 1
    int Br = std::min(Bc, d);

    // 在 host 上分配并初始化数据(如 Q,K,V,m,l,O)
    float *hQ = new float[N*d];     // Q
    float *hK = new float[N*d];     // K
    float *hV = new float[N*d];     // V
    float *hO = new float[N*d];     // O
    float *hM = new float[N];       // M 每行（d维度）最大的值
    float *hL = new float[N];       // L sum(exp(⋯))

    // 简单初始化
    for(int i = 0; i < N*d; ++i) {
        hQ[i] = 0.01f * (i % d);
        hK[i] = 0.02f * (i % d);
        hV[i] = 0.03f * (i % d);
        hO[i] = 0.f;
    }

    for(int i=0; i<N; ++i){
        hM[i] = -1e30f; // -∞
        hL[i] = 0.f;
    }

    // 分配 GPU 内存
    float *dQ, *dK, *dV, *dO, *dM, *dL;
    CUDA_CHECK(cudaMalloc(&dQ, N*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK, N*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV, N*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dO, N*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dM, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dL, N*sizeof(float)));

    // 拷贝到 GPU
    CUDA_CHECK(cudaMemcpy(dQ, hQ, N*d*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK, N*d*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV, N*d*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dO, hO, N*d*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dM, hM, N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dL, hL, N*sizeof(float), cudaMemcpyHostToDevice));

    // 调用 FlashAttention 前向
    printf("flashAttentionFwd_Inplace...\n");
    flashAttentionFwd_Inplace(dQ, dK, dV, dO, dM, dL, N, d, Br, Bc);

    // 把结果拷回主机
    CUDA_CHECK(cudaMemcpy(hO, dO, N*d*sizeof(float), cudaMemcpyDeviceToHost));

    const std::string filename = "ref_attention.txt";
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<std::vector<float>> attention_result;
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::vector<float> row;
        float val;
        while (iss >> val) {
            row.push_back(val);
        }
        if (!row.empty()) {
            attention_result.push_back(row);
        }
    }
    infile.close();

    // 输出文件中前几行的数值进行验证
    std::cout << attention_result.size() << "  " << attention_result[0].size() << std::endl;
    for (int i = 0; i < attention_result.size(); ++i) {
        for (int j = 0; j < attention_result[i].size(); ++j) {
            if(attention_result[i][j] - hO[i*d+j] > 1e-3) {
                printf("not match in (%d, %d) %f vs %f\n",i, j, attention_result[i][j], hO[i*d+j]);
                break;
            }
        }
    }


    // 简单打印部分结果
    for(int i=0; i<5; ++i){
        printf("O[%d]=%f ", i, hO[i]);
    }
    printf("...\n");

    // 清理
    delete[] hQ; delete[] hK; delete[] hV; delete[] hO; delete[] hM; delete[] hL;
    CUDA_CHECK(cudaFree(dQ)); CUDA_CHECK(cudaFree(dK)); 
    CUDA_CHECK(cudaFree(dV)); CUDA_CHECK(cudaFree(dO));
    CUDA_CHECK(cudaFree(dM)); CUDA_CHECK(cudaFree(dL));

    return 0;
}
