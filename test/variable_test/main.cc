#include "Variable.h"
#include <iostream>
#include <vector>

int main() {
    // 初始化一些数据
    float tensor_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    std::vector<int> tensor_shape = {3, 4}; // 3x4矩阵

    float parameter_data[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
    std::vector<int> parameter_shape = {3, 3}; // 3x3矩阵

    // 创建一个 Tensor 对象
    Tensor tensor1("input_tensor", tensor_data, tensor_shape, "CPU");
    tensor1.whereami();
    std::cout << "Tensor Name: " << tensor1.Name() << "\n";
    std::cout << "Tensor Size: " << tensor1.Size() << "\n\n";
    
    // 将 Tensor 从 CPU 传输到 GPU
    tensor1.to("GPU");
    tensor1.whereami();
    std::cout << "Tensor Name: " << tensor1.Name() << "\n";
    std::cout << "Tensor Size: " << tensor1.Size() << "\n\n";

    // 创建一个 Parameter 对象
    Parameter param1("weight_param", parameter_data, parameter_shape, "CPU");
    param1.whereami();
    std::cout << "Parameter Name: " << param1.Name() << "\n";
    std::cout << "Parameter Size: " << param1.Size() << "\n\n";

    // 将 Parameter 从 CPU 传输到 GPU
    param1.to("GPU");
    param1.whereami();
    std::cout << "Parameter Name: " << param1.Name() << "\n";
    std::cout << "Parameter Size: " << param1.Size() << "\n\n";

    return 0;
}
