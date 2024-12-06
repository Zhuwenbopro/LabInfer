#include "CPUFunction.h"
#include <cmath>
#include <string.h>

void embedding_cpu(float* y, const float* x, const float* W, const int d, const int x_size) {
    for(int i = 0; i < x_size; i++) {
        int id = (int)x[i];
        memcpy(y + i * d, W + id * d, sizeof(float) * d);
    }
}
void embedding_cpu(float**y, float*x, float*W, int num, int hidden_size) {
    for(int i = 0; i < num; i++) {
        int id = (int)x[i];
        memcpy(y[i], W + id * hidden_size, sizeof(float) * hidden_size);
    }
}

void matmul_cpu(float **y, float **x, float *W, int n, int d, int num) {
    for(int no = 0; no < num; no++) {
        float *y_ptr = y[no];
        const float *x_ptr = x[no];
        for(int i = 0; i < d; i++) {
            double sum = 0.0f;
            for(int j = 0; j < n; j++) {
                sum += W[i * n + j] * x_ptr[j];
            }
            y_ptr[i] = sum;
        }
    }
}
void matmul_cpu(float *y, const float *x, const float *w, int n, int d, int num) {
    for(int b = 0; b < num; b++){
        for (int i = 0; i < d; ++i) {
            double sum = 0.0f;
            for (int j = 0; j < n; ++j) {    
                sum += w[i * n + j] * x[b * n + j];
            }
            y[b*d + i] = sum;
        }
    }
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
    for(int b = 0; b < batch_size; b++) {
        // 找到输入数组中的最大值，以提高数值稳定性
        float* input = x + b * n;
        float max_val = input[0];
        for(int i = 1; i < n; ++i){
            if(input[i] > max_val){
                max_val = input[i];
            }
        }

        // 计算每个元素的指数值，并累加
        float sum = 0.0f;
        for(int i = 0; i < n; ++i){
            input[i] = std::exp(input[i] - max_val);
            sum += input[i];
        }

        // 将每个指数值除以总和，得到概率分布
        for(int i = 0; i < n; ++i){
            input[i] /= sum;
        }
    }
}
void softmax_cpu(float**x, int n, int num) {
    for(int i = 0; i < num; i++) {
        float* x_ptr = x[i];
        float max_val = x_ptr[0];
        for(int i = 1; i < n; ++i) {
            if(x_ptr[i] > max_val){
                max_val = x_ptr[i];
            }
        }

        // 计算每个元素的指数值，并累加
        float sum = 0.0f;
        for(int i = 0; i < n; ++i) {
            x_ptr[i] = std::exp(x_ptr[i] - max_val);
            sum += x_ptr[i];
        }

        // 将每个指数值除以总和，得到概率分布
        for(int i = 0; i < n; ++i) {
            x_ptr[i] /= sum;
        }
    }
}


// dim = 32, num = 6
void apply_rope_cpu(float *_x, const float *_pos, const float *_cos, const float *_sin, const int n, const int dim, const int num) {
    for(int p = 0; p < num; p++){   // 6
        const float* cos = _cos + (int)_pos[p] * dim;
        const float* sin = _sin + (int)_pos[p] * dim;
        for(int i = 0; i < n/(dim*2); i++) {
            float* x = _x + p*n + i*dim*2; 
            for(int j = 0; j < dim; j++) {
                float x1 = x[j];
                float x2 = x[dim + j];
                x[j]       = x1 * cos[j] - x2 * sin[j];
                x[dim + j] = x2 * cos[j] + x1 * sin[j];
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
void silu_cpu(float **x, int n, int num){
    for(int i = 0; i < num; i++){
        float* x_ptr = x[i];
        for(int i = 0; i < n; i++){
            x_ptr[i] = x_ptr[i] / (1 + std::exp(-x_ptr[i]));
        }
    }
}


void add_cpu(float* y, const float* x1, const float* x2, const int n, int batch_size) {
    for(int b = 0; b < batch_size; b++){
        for(int i = 0; i < n; i++) {
            y[i + b*n] = x1[i + b*n] + x2[i + b*n];
        }
    }
}
void add_cpu(float**y, float**x1, float**x2, int n, int num) {
    for(int i = 0; i < num; i++){
        float* y_ptr = y[i];
        float* x1_ptr = x1[i];
        float* x2_ptr = x2[i];
        for(int i = 0; i < n; i++){
            y_ptr[i] = x1_ptr[i] + x2_ptr[i];
        }
    }
}

// query        (1, dim*head_num)
// key、value   (pos, dim*k_num)
// y            (1, dim*head_num)
// pos 是 query 的 position
void maksed_attention_cpu(float* y, const float* q, const float* k, const float* v, const int dim, const int q_head, const int kv_head, const int _pos) {
    int pos = _pos + 1;
    float* score = new float[q_head * pos](); // 置初始值为0，列优先，pos行，q_head列

    int rep = q_head / kv_head;
    int kv_dim = kv_head * dim;

    float scale = 1.0 / std::sqrt(static_cast<float>(dim));
    for(int p = 0; p < pos; p++) {
        for(int hq = 0; hq < q_head; hq++) {
            const float* _q = q + hq * dim;
            const float* _k = k + p * kv_dim + (hq / rep) * dim;
            const int s_index = hq*pos + p;
            for(int d = 0; d < dim; d++) {
                score[s_index] += _q[d] * _k[d];
            }
            score[s_index] *= scale;
        }
    }

    softmax_cpu(score, pos, q_head);

    std::memset(y, 0, dim * q_head * sizeof(float));

    for(int hq = 0; hq < q_head; hq++) {
        float* _s = score + hq * pos;
        float* _y = y + hq * dim;
        for(int p = 0; p < pos; p++) {
            const float* _v = v + p * kv_dim + (hq / rep) * dim;
            for(int d = 0; d < dim; d++) {
                _y[d] += _s[p] * _v[d];
            }
        }
    }

    delete score;
}

void elem_multiply_cpu(float* y, const float* x1, const float* x2, const int size) {
    for(int i = 0; i < size; i++) {
        y[i] = x1[i] * x2[i];
    }
}
void elem_multiply_cpu(float**y, float**x1, float**x2, int n, int num) {
    for(int i = 0; i < num; i++){
        float* y_ptr = y[i];
        float* x1_ptr = x1[i];
        float* x2_ptr = x2[i];
        for(int i = 0; i < n; i++){
            y_ptr[i] = x1_ptr[i] * x2_ptr[i];
        }
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
void max_index_cpu(float* index, float** x, int n, int num) {
    for (int i = 0; i < num; i++) {
        int max_idx = 0;
        float max_val = x[i][0];
        for (int j = 1; j < n; j++) {
            if (x[i][j] > max_val) {
                max_val = x[i][j];
                max_idx = j;
            }
        }
        index[i] = (float)max_idx;
    }
}