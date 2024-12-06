// CudaFunctionLibrary.h
#ifndef CPU_FUNCTION_H
#define CPU_FUNCTION_H

#include "Function.h"
#include <cstring>

void matmul_cpu(float *y, const float *x, const float *w, int n, int d, int num);
void matmul_cpu(float **y, float **x, float *W, int n, int d, int num);

void rmsnorm_cpu(float* x, const float* w, int n, int batch_size, const float epsilon);

void softmax_cpu(float *x, int n, int batch_size);
void softmax_cpu(float**x, int n, int num);

void apply_rope_cpu(float *_x, const float *pos, const float *_cos, const float *_sin, const int n, const int dim, const int num);

void silu_cpu(float *x, const int n, int batch_size);
void silu_cpu(float**x, int n, int num);

void add_cpu(float* y, const float* x1, const float* x2, const int n, int batch_size);
void add_cpu(float**y, float**x1, float**x2, int n, int num);

void embedding_cpu(float* y, const float* x, const float* W, const int d, const int x_size);
void embedding_cpu(float**y, float*x, float*W, int num, int hidden_size);

void maksed_attention_cpu(float* y, const float* q, const float* k, const float* v, const int dim, const int q_head, const int kv_head, const int _pos);

void elem_multiply_cpu(float* y, const float* x1, const float* x2, const int size);
void elem_multiply_cpu(float**y, float**x1, float**x2, int n, int num);

void max_index_cpu(float* index, float* x, const int n, const int num);
void max_index_cpu(float* index, float** x, int n, int num);

class CPUFunction : public Function {
    void whereami() override {
        std::cout << "Function in CPU" << std::endl;
    }

    void matmul(float *y, const float *x, const float *w, const int n, const int d, const int num = 1) override {
        matmul_cpu(y, x, w, n, d, num);
    }
    void matmul(float**y, float**x, float* W, int n, int d, int num) override {
        matmul_cpu(y, x, W, n, d, num);
    }

    void rmsnorm(float* x, const float* w, const int n, const int batch_size = 1, const float epsilon=1e-5) override {
        rmsnorm_cpu(x, w, n, batch_size, epsilon);
    }

    void softmax(float *x, const int n, const int batch_size = 1) override {
        softmax_cpu(x, n, batch_size);
    }
    void softmax(float **x, int n, int num) override {
        softmax_cpu(x, n, num);
    }

    void apply_rope(float *x, const float *pos, const float *cos, const float *sin, const int n, const int dim, const int num) override {
        apply_rope_cpu(x, pos, cos, sin, n, dim, num);
    }

    void silu(float *x, const int n, const int batch_size = 1) override {
        silu_cpu(x, n, batch_size);
    }
    void silu(float **x, int n, int num) override {
        silu_cpu(x, n, num);
    }

    void add(float* y, const float* x1, const float* x2, const int n, const int batch_size = 1) override {
        add_cpu(y, x1, x2, n, batch_size);
    }
    void add(float**y, float**x1, float**x2, int n, int num) override {
        add_cpu(y, x1, x2, n, num);
    }

    void embedding(float* y, const float* x, const float* W, const int d, const int x_size) override {
        embedding_cpu(y, x, W, d, x_size);
    }
    void embedding(float**y, float* x, float* W, int num, int hidden_size) override {
        embedding_cpu(y, x, W, num, hidden_size);
    }

    void maksed_attention(float* y, const float* q, const float* k, const float* v, const int dim, const int q_head, const int kv_head, const int _pos) override {
        maksed_attention_cpu(y, q, k, v, dim, q_head, kv_head, _pos);
    }

    void elem_multiply(float* y, const float* x1, const float* x2, const int size) override {
        elem_multiply_cpu(y, x1, x2, size);
    }
    void elem_multiply(float**y, float**x1, float**x2, int n, int num) override {
        elem_multiply_cpu(y, x1, x2, n, num);
    }

    void max_index(float* index, float* x, const int n, const int num) override {
        max_index_cpu(index, x, n, num);
    }
    void max_index(float* index, float** x, int n, int num) override {
        max_index_cpu(index, x, n, num);
    }
};
#endif // CPU_FUNCTION_H
