#include "Tensor.h"
#include <iostream>
#include <vector>

int main() {
    // 创建输入数据
    std::vector<std::vector<size_t>> input_ids = {
        {1, 2, 3},
        {4, 5, 6, 7},
        {8, 9}
    };
    std::vector<size_t> uid = {12231, 44223, 5555};

    // 设备（假设 "cpu" 是有效的设备）
    std::string device = "cpu";

    // 创建一个 Tensor 对象
    Tensor tensor(input_ids, device);
    tensor.setUid(uid);
    tensor.addPos(input_ids);

    // 输出一些信息
    std::cout << "Tensor Size: " << tensor.Size() << std::endl;
    std::cout << "Element Length: " << tensor.elemLen() << std::endl;
    std::cout << "Element Number: " << tensor.elemNum() << std::endl;
    std::cout << "Device: " << tensor.Device() << std::endl;
    std::cout << "SeqLen : " << tensor.SeqLen()[0] << "  " << tensor.SeqLen()[1] << "  " << tensor.SeqLen()[2] << std::endl;

    Tensor t = tensor.tail();
    std::cout << "tensor tail : ";
    for(int i = 0; i < t.elemNum(); i++)
        std::cout << t[i] << " ";
    std::cout << std::endl;

    std::cout << "tensor tail pos : ";
    for(int i = 0; i < t.elemNum(); i++)
        std::cout << t.Position()[i][0] << " ";
    std::cout << std::endl;

    // 访问原始数据
    float* data = tensor.rawPtr();

    // 打印数据
    std::cout << "Tensor Data: ";
    for (size_t i = 0; i < tensor.Size(); ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;

    // 创建另一个相同大小的 Tensor 用于复制
    std::vector<std::vector<size_t>> input_ids2 = {
        {10, 11, 12},
        {13, 14, 15, 16},
        {17, 18}
    };

    Tensor tensor2(input_ids2, device);

    // 将 tensor2 复制到 tensor
    try {
        tensor.copy(tensor2);

        // 复制后打印数据
        std::cout << "Tensor Data after copy: ";
        data = tensor.rawPtr();
        for (size_t i = 0; i < tensor.Size(); ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    } catch (const std::logic_error& e) {
        std::cerr << "复制失败: " << e.what() << std::endl;
    }

    // 测试 to() 函数，将设备转换为 "gpu"
    tensor.to("cuda");
    std::cout << "Tensor Device after to(): " << tensor.Device() << std::endl;

    return 0;
}
