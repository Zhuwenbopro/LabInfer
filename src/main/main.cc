#include <iostream>

// 声明外部函数
extern "C" void addVectorsWrapper(const float* a, const float* b, float* c, int n);

int main() {
    int n = 1000;
    float *a = new float[n];
    float *b = new float[n];
    float *c = new float[n];

    // 初始化输入数据
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }

    // 调用动态库中的函数
    addVectorsWrapper(a, b, c, n);

    // 验证结果
    bool success = true;
    for (int i = 0; i < n; ++i) {
        if (c[i] != a[i] + b[i]) {
            success = false;
            std::cout << "错误在索引 " << i << ": " << c[i] << " != " << a[i] + b[i] << std::endl;
            break;
        }
    }

    if (success)
        std::cout << "计算成功！" << std::endl;

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}

