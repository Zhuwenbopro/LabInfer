#include "Tensor.h"
#include "Variable.h"
#include "../../src/layer/Parameter.h"
#include <unordered_map>

int main() {
    // 创建一个简单的 float 数组用于测试
    size_t size_in = 2;
    size_t size_out = 3;
    float data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    std::cout << "===== Tensor Test Started =====\n";
    
    // 1. 测试构造函数
    std::cout << "Testing Tensor constructor...\n";
    Tensor tensor1("W", data, {size_in, size_out}, "cpu");
    
    // 打印初始值
    std::cout << "Tensor name: " << tensor1.Name() << "\n";
    std::cout << "Tensor shape: ";
    for (size_t dim : tensor1.Shape()) {
        std::cout << dim << " ";
    }
    std::cout << "\n";
    std::cout << "Tensor device: " << tensor1.Device() << "\n";

    // 2. 浅拷贝测试
    std::cout << "\nTesting shallow copy...\n";
    Tensor tensor2 = tensor1;  // 浅拷贝
    std::cout << "Shallow copied tensor name: " << tensor2.Name() << "\n";
    std::cout << "Shallow copied tensor device: " << tensor2.Device() << "\n";

    // 修改 tensor1 的值，看看 tensor2 是否受到影响（浅拷贝应该会受影响）
    data[0] += 4;
    std::cout << "After modifying tensor1, tensor2 data[0]: " << tensor2.Data()[0] << "\n";

    // 3. 深拷贝测试
    std::cout << "\nTesting deep copy...\n";
    Tensor tensor3 = tensor1.copy();  // 深拷贝
    std::cout << "Deep copied tensor name: " << tensor3.Name() << "\n";
    std::cout << "Deep copied tensor device: " << tensor3.Device() << "\n";

    // 修改 tensor1 的值，看看 tensor3 是否受到影响（深拷贝不应该受影响）
    data[0] += 4;
    std::cout << "After modifying tensor1, tensor3 data[0]: " << tensor3.Data()[0] << "\n";

    // 4. 设备传输测试
    std::cout << "\nTesting device transfer...\n";
    tensor3.to("cuda");  // 将 tensor1 从 CPU 转移到 GPU
    std::cout << "After transferring, tensor3 device: " << tensor3.Device() << "\n";

    // 确保没有影响原来的 tensor1（浅拷贝会影响）
    std::cout << "Tensor1 device (after tensor1 transfer): " << tensor1.Device() << "\n";

    std::cout << "===== Tensor Test Completed =====\n";


    std::string name = "TestParameter";
    std::string device = "cpu"; // Initial device

    // 创建 Parameter 对象
    Parameter param(name, data, {2, 3}, device);

    // 验证参数值
    std::cout << "Parameter name: " << param.Name() << std::endl;
    std::cout << "Parameter device: " << param.Device() << std::endl;
    std::cout << "Parameter size: " << param.Size() << std::endl;
    
    // 打印形状
    std::cout << "Parameter shape: [";
    for (size_t i = 0; i < param.Shape().size(); ++i) {
        std::cout << param.Shape()[i] << (i < param.Shape().size() - 1 ? ", " : "");
    }
    std::cout << "]" << std::endl;

    std::unordered_map<std::string, Parameter> params;
    params.emplace("W", Parameter("W", nullptr, {size_in, size_out}, "cpu"));
}