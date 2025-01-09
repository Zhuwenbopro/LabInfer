#ifndef Function_H
#define Function_H
#include <iostream>

class Function {
public:

    virtual void whereami() = 0;

    /**
    * @brief 对输入数据执行均方根归一化（RMS Normalization）。
    *
    * 此虚函数用于对输入数组进行均方根归一化处理。归一化过程考虑了权重数组和一个小的 epsilon 值，以确保数值稳定性并防止除零错误。
    * 归一化后的结果将被存储在输出数组中。具体的实现由派生类提供。
    *
    * @param[out] y 指向用于存储归一化后数据的输出数组的指针。数组大小应至少为 `n`。
    * @param[in] x 指向要进行归一化处理的输入数据数组的常量指针。数组大小应至少为 `n`。
    * @param[in] w 指向用于归一化的权重数组的常量指针。数组大小应至少为 `n`。
    * @param n 要处理的向量维度。
    * @param[in] batch_size 输入批次。默认为 1。
    * @param epsilon 一个小的浮点数值，用于在计算过程中防止除零错误或数值不稳定性。通常取一个非常小的值，如 `1e-5`。
    * 
    */
    virtual void rmsnorm(float* x, const float* w, int n, int batch_size = 1, const float epsilon=1e-5) = 0;

    /**
    * @brief 矩阵乘法操作。batch x (1 x n) x (n x d)
    *
    * 矩阵 W 是列优先写的，一条一条写
    *
    * @param[out] y 指向用于存储结果矩阵的输出数组的指针。数组大小应至少为 `num * d`。
    * @param[in] x 矩阵的尺寸为 `num x n`。
    * @param[in] W 矩阵的尺寸为 `n x d`。
    * @param[in] n 输入向量 `x` 的维度。
    * @param[in] d 输入矩阵 `w` 的列数。
    * @param[in] num 有几个 x。
    */
    virtual void matmul(float* y, const float *x, const float *W, const int n, const int d, const int num) = 0;

    virtual void softmax(float* x, const int n, const int num) = 0;

    /**
    * @brief 对 x 做位置编码
    *
    *
    * @param[out] x 一维向量 长度为 `n * num`。
    * @param[in] pos x 的位置，长度为 `1 x num`。
    * @param[in] cos 提前算好的位置余弦 `max_pos * dim`。
    * @param[in] n 输入向量 `x` 的维度。
    * @param[in] head_dim = head_dim
    * @param[in] num 有多少个 pos
    */
    virtual void apply_rope(float *x, const int *pos, const float *inv_freq, const int n, int head_dim, const int num) = 0;

    virtual void silu(float* x, const int n, const int batch_size = 1) = 0;

    virtual void add(float* y, const float* x1, const float* x2, const int n, const int batch_size = 1) = 0;

    /**
    * @brief [batch, seq] => [batch, seq, dim]
    *
    * 不管怎么样，多少维度，数据怎么存储有个窍门：你从最后的维度向前看，依次存储。
    * 比如说：[2,3,4] 这样的矩阵在数组中存储是 [[4] [4] [4]] [[4] [4] [4]]
    *
    * @param[out] y 指向用于存储结果矩阵的输出数组的指针。数组大小应至少为 `b x s x d`。
    * @param[in] x 指向第一个输入向量的常量指针。矩阵的尺寸为 `b x s`。
    * @param[in] W 指向第二个输入矩阵的常量指针。矩阵的尺寸为 `vocab x s`。
    * @param[in] d 输出单词的词向量 token tensor 维度。
    * @param[in] x_size x 输入 token 数。
    * @param[in] batch_size 输入批次。默认为 1。
    */
    virtual void embedding(float* y, const int* x, const float* W, const int d, const int x_size) = 0;

    /**
     *  @brief 
     * 
     *  no matter which device it use, _pos in cpu
     */
    virtual void masked_attention(float* y, float* q, float* k, float* v, float* scores, int* pos, int dim, int head_num, int seq_q, int seq_kv) = 0;

    virtual void elem_multiply(float* y, const float* x1, const float* x2, const int size) = 0;

    virtual void max_index(float* index, float* x, const int n, const int num) = 0;
    
    // n 是 in 的元素长度
    virtual void repeat_kv(float* o, float* in, int dim, int rep, int n) = 0;
};

#endif // Function_H