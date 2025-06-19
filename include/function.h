#pragma once

using LinearFuncPtr = void (*)(void *y, void *X, void *W, int W_in, int W_out, int num);

using RoPEFuncPtr = void (*)(void *x, int *pos, void *inv_freq, int n, int head_dim, int head_num);

using SoftmaxFuncPtr = void (*)(void *x, int n, int num);

using SiluFuncPtr = void (*)(void *x, int n, int num);

using MultiplyElementWiseFuncPtr = void (*)(void *y, void *x, int n, int num);

using AddElementWiseFuncPtr = void (*)(void *y, void *x, int n, int num);

using RMSNormFuncPtr = void (*)(void *x, void *w, int n, int num, float e);

using MaxIndexFuncPtr = void (*)(int *index, void *x, int n, int num);

// // Attention核心计算函数指针
// using AttentionFuncPtr = void (*)(
//     const Tensor& q,
//     const Tensor& k,
//     const Tensor& v,
//     const Tensor* mask, // 可选的mask
//     Tensor& output
// );
