#include "CPUFunction.h"
#include <cmath>
#include <string.h>
#include <omp.h>
#include <cblas.h>
#include <vector>

inline float dot(float* a, float* b, size_t size) {
    return cblas_sdot(size, a, 1, b, 1);
}

inline void scale(float* a, float alpha, size_t size) {
    cblas_sscal(size, alpha, a, 1);
}

// Y = alpha * X + Y
inline void _add(float* y, float* x, size_t size, float alpha = 1) {
    cblas_saxpy(size, alpha, x, 1, y, 1);
}


void embedding_cpu(float* y, const int* x, const float* W, const int d, const int x_size) {
    #pragma omp parallel for
    for(int i = 0; i < x_size; i++) {
        int id = x[i];
        memcpy(y + i * d, W + id * d, sizeof(float) * d);
    }
}

// y = WX     W(W_in*W_out), X(W_in*num), C(W_out*num)  
void matmul_cpu(float *y, const float *X, const float *W, int W_in, int W_out, int num) {
    // 缩放因子
    float alpha = 1.0;
    float beta = 0.0;  // C 的初始权重
    // 调用 OpenBLAS 的 SGEMM 函数
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                W_out, num, W_in,         // 矩阵维度
                alpha,           // alpha
                W, W_in,            // 矩阵 W 和列主布局步长
                X, W_in,            // 矩阵 X 和列主布局步长
                beta,            // beta
                y, W_out);           // 结果矩阵 C 和列主布局步长
}



void rmsnorm_cpu(float* x, const float* w, int n, int batch_size, const float epsilon) {
    for(int b = 0; b < batch_size; b++) {
        // 求平方和
        float sum_of_squares = 0.0f;
        for (int i = 0; i < n; ++i) {
            int index = i + b * n;
            sum_of_squares += x[index] * x[index];
        }

        // 计算均方根归一化系数
        float mean_square = sum_of_squares / n;
        float rms = 1.0f / std::sqrt(mean_square + epsilon); // 防止除以零

        // 归一化并乘以权重
        for (int i = 0; i < n; ++i) {
            int index = i + b * n;
            x[index] = w[i] * x[index] * rms;
        }
    }
}

void softmax_cpu(float *x, int n, int batch_size) {
    // Step 1: Subtract max value from each column (vector) for numerical stability
    #pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
        // Find the maximum element in the column
        float max_val = -MAXFLOAT;
        for (int j = 0; j < n; ++j) {
            int idx = i * n + j; // Column-major index calculation
            max_val = std::max(max_val, x[idx]);
        }
        
        // Subtract the max from each element in the column
        for (int j = 0; j < n; ++j) {
            int idx = i * n + j; // Column-major index calculation
            x[idx] -= max_val;
        }
    }

    // Step 2: Compute the exponential of each element
    #pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < n; ++j) {
            int idx = i * n + j;
            x[idx] = std::exp(x[idx]); // Element-wise exp
        }
    }

    // Step 3: Compute the sum of exponentials for each column
    std::vector<float> row_sums(batch_size, 0.0f);
    #pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            int idx = i * n + j;
            sum += x[idx];
        }
        row_sums[i] = sum;
    }

    // Step 4: Normalize each element by the column sum
    #pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
        float row_sum = row_sums[i];
        for (int j = 0; j < n; ++j) {
            int idx = i * n + j;
            x[idx] /= row_sum;
        }
    }
}

// dim = 32, num = 6
void apply_rope_cpu(
    float *x, 
    const int *pos, 
    const float *inv_freq, 
    const int n,            // x 有 n 维
    int head_dim,           // x 每一小段有 head_dim 维
    const int num           // x 有 num 个
) {
    const int loop_count = n / head_dim;
    int dim = head_dim / 2;

    #pragma omp parallel for
    for(int p = 0; p < num; p++) {

        int pos_idx = pos[p];
        float* xBase = x + p*n;

        for(int j = 0; j < dim; j++) {
            float c = cosf(pos_idx*inv_freq[j]);
            float s = sinf(pos_idx*inv_freq[j]);
            for(int i = 0; i < loop_count; i++) {
                float x1 = xBase[i*head_dim + j];
                float x2 = xBase[i*head_dim + j + dim];
                xBase[i*head_dim + j]       = x1 * c - x2 * s;
                xBase[i*head_dim + j + dim] = x2 * c + x1 * s;
            }
        }
    }
}

void silu_cpu(float *x, const int n, int batch_size){
    for(int b = 0; b < batch_size; b++){
        float* input = b*n + x;
        for(int i = 0; i < n; i++){
            input[i] = input[i] / (1 + std::exp(-input[i]));
        }
    }
}

void add_cpu(float* y, const float* x1, const float* x2, const int n, int batch_size) {
    size_t total = n*batch_size;
    for(int i = 0; i < total; i++) {
        y[i] = x1[i] + x2[i];
    }
}

void repeat_kv_cpu(float* out, float* in, int dim, int rep, int n) {
    int loop = n / dim;
    for(int l = 0; l < loop; l++) {
        float* in_base = in + l*dim;
        float* out_base = out + l*dim*rep;
        for(int r = 0; r < rep; r++) {
            for(int d = 0; d < dim; d++) {
                out_base[r*dim+ d] = in_base[d];
            }
        }
    }
}

// query        (1, dim*head_num)
// key、value   (pos, dim*k_num)
// y            (1, dim*head_num)
// pos 是 query 的 position
// (float* y, float* q, float* k, float* v, float* scores, int* pos, int dim, int head_num, int seq_q, int seq_kv) = 0;
void CPUFunction::masked_attention(float* y, float* q, float* k, float* v, float* scores, int* pos, int dim, int head_num, int seq_q, int seq_kv) {
    bool hasvalue = true;
    if(scores == nullptr) {
        scores = new float[seq_kv * head_num * seq_q];
        hasvalue = false;
    }

    std::memset(y, 0, dim * head_num * seq_q * sizeof(float));

    float scale_ = 1.0 / std::sqrt(static_cast<float>(dim));

    int kv_num_ = seq_kv - seq_q;
    for(int i_q = 0; i_q < seq_q; i_q++) {
        kv_num_++;
        float* q_ = q + i_q * dim * head_num;
        float* y_ = y + i_q * dim * head_num;
        for(int i_kv = 0; i_kv < kv_num_; i_kv++) {
            float* k_ = k + i_kv * dim * head_num;
            for(int h = 0; h < head_num; h++) {
                scores[i_kv + h*kv_num_] = dot(q_ + h*dim, k_ + h*dim, dim);
            }
        }
        scale(scores, scale_, kv_num_*head_num);
        softmax_cpu(scores, kv_num_, head_num);

        for(int i_kv = 0; i_kv < kv_num_; i_kv++) {
            float* v_ = v + i_kv * dim * head_num;
            for(int h = 0; h < head_num; h++) {
                _add(y_ + h*dim, v_ + h*dim, dim, scores[i_kv + h*kv_num_]);
            }
        }
    }

    if(!hasvalue) delete scores;
}

void elem_multiply_cpu(float* y, const float* x1, const float* x2, const int size) {
    for(int i = 0; i < size; i++) {
        y[i] = x1[i] * x2[i];
    }
}

void max_index_cpu(float* index, float* x, const int n, const int num) {
    for (int i = 0; i < num; i++) {
        int max_idx = 0;
        float max_val = x[i * n];
        for (int j = 1; j < n; j++) {
            if (x[i * n + j] > max_val) {
                max_val = x[i * n + j];
                max_idx = j;
            }
        }
        index[i] = (float)max_idx;
    }
}
