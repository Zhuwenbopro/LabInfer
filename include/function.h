#pragma once

using LinearFuncPtr = void (*)(void *y, void *X, void *W, int W_in, int W_out, int num);

// // Attention核心计算函数指针
// using AttentionFuncPtr = void (*)(
//     const Tensor& q,
//     const Tensor& k,
//     const Tensor& v,
//     const Tensor* mask, // 可选的mask
//     Tensor& output
// );

// // RMS Norm计算函数指针
// using RMSNormFuncPtr = void (*)(
//     const Tensor& input,
//     const Tensor& weight,
//     float epsilon,
//     Tensor& output
// );