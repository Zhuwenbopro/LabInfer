#include "registry.h"
#include <cblas.h>

// y = WX     W(W_in*W_out), X(W_in*num), C(W_out*num)  
void cpu_fp32_linear_exec(void *y, void *X, void *W, int W_in, int W_out, int num)
{
    // 缩放因子
    float alpha = 1.0;
    float beta = 0.0;  // C 的初始权重
    // 调用 OpenBLAS 的 SGEMM 函数
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                (float *)W_out, num, (float *)W_in,          // 矩阵维度
                alpha,
                (float *)W, W_in,                            // 矩阵 W 和列主布局步长
                (float *)X, W_in,                            // 矩阵 X 和列主布局步长
                beta,            // beta
                (float *)y, (float *)W_out);                 // 结果矩阵 C 和列主布局步长
}

REGISTER_OP_FUNCTION(Linear, CPU, FLOAT32, cpu_fp32_linear_exec);