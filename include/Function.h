#ifndef Function_H
#define Function_H


class Function {
public:

    /**
    * @brief 对输入数据执行均方根归一化（RMS Normalization）。
    *
    * 此虚函数用于对输入数组进行均方根归一化处理。归一化过程考虑了权重数组和一个小的 epsilon 值，以确保数值稳定性并防止除零错误。
    * 归一化后的结果将被存储在输出数组中。具体的实现由派生类提供。
    *
    * @param output 指向用于存储归一化后数据的输出数组的指针。数组大小应至少为 `size`。
    * @param input 指向要进行归一化处理的输入数据数组的常量指针。数组大小应至少为 `size`。
    * @param weight 指向用于归一化的权重数组的常量指针。数组大小应至少为 `size`。
    * @param epsilon 一个小的浮点数值，用于在计算过程中防止除零错误或数值不稳定性。通常取一个非常小的值，如 `1e-5`。
    * @param size 要处理的数据元素的数量。`input`、`output` 和 `weight` 数组应至少包含 `size` 个元素。
    *
    */
    virtual void rmsnorm(float* output, const float* input, const float* weight, const float epsilon, int size) = 0;

    /**
    * @brief 矩阵乘法操作。(目前还只是 一维向量 * 矩阵)
    *
    * 此纯虚函数用于在 CPU 上对两个输入矩阵进行矩阵乘法运算。计算结果存储在输出矩阵中。具体的实现由派生类提供。
    *
    * @param[out] xout 指向用于存储结果矩阵的输出数组的指针。数组大小应至少为 `n * d`。
    * @param[in] x 指向第一个输入矩阵的常量指针。矩阵的尺寸为 `n x d`。数组大小应至少为 `n * d`。
    * @param[in] w 指向第二个输入矩阵的常量指针。矩阵的尺寸为 `d x m`（假设 `m` 为另一维度，具体取决于实现）。数组大小应至少为 `d * m`。
    * @param[in] n 输入矩阵 `x` 的行数。
    * @param[in] d 输入矩阵 `x` 和 `w` 的列数，也是 `w` 的行数。
    *
    * @warning 确保 `xout`、`x` 和 `w` 指针指向的内存区域已正确分配，并且大小满足上述要求，以避免内存访问错误。
    */
    virtual void matmul(float* xout, const float *x, const float *w, int n, int d) = 0;

    virtual void softmax(float* x, int n) = 0;

    virtual void rotary_positional_embedding(int pos, float *vec, int dim, int head_size) = 0;

    virtual void silu(float* x, const int n) = 0;

    virtual void add(float* y, const float* x1, const float* x2, const int n) = 0;
};

#endif // Function_H