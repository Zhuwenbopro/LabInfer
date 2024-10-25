#include "Tensor.h"
#include "Variable.h"
#include "../../src/layer/Parameter.h"
#include <unordered_map>

int main() {
    // 创建一个简单的 float 数组用于测试
    // 测试 Tensor 类
    std::vector<size_t> tensor_shape = {2, 2, 3}; // 2x3 的张量
    std::vector<size_t> seq_shape = {1, 2};    // 序列形状
    Tensor tensor("tensor1", tensor_shape, "cpu", true, seq_shape);
    std::cout << "Tensor created with shape (2, 2, 3) on device: " << tensor.Device() << std::endl;
    std::cout << "Tensor size:    " << tensor.Size() << std::endl;
    std::cout << "Tensor elenNum: " << tensor.elemNum() << std::endl;
    std::cout << "Tensor elenLen: " << tensor.elemLen() << std::endl;

    Tensor deep_copy = tensor.copy();
    tensor.to("cuda");
    Tensor shallow_copy = tensor;
    std::cout << "tensor on device: " << tensor.Device() << std::endl;
    std::cout << "Shallow copied tensor created with shape (2, 3) on device: " << shallow_copy.Device() << std::endl;
    std::cout << "Deep copied tensor created with shape (2, 3) on device: " << deep_copy.Device() << std::endl;


    // 测试 Parameter 类
    std::vector<size_t> param_shape = {1, 10}; // 1x10 的参数
    Parameter param("param1", param_shape, "cpu", true);
    std::cout << "Parameter created with shape (1, 10) on device: " << param.Device() << std::endl;
    
    // 测试设备间传输（对于 Parameter）
    param.to("cuda");
    std::cout << "Parameter moved to device: " << param.Device() << std::endl;

    // 测试共享参数设备传输
    param.setShared();
    param.to("cuda");
    std::cout << "Shared parameter moved to device: " << param.Device() << std::endl;

}