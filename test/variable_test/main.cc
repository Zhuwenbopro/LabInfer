#include "Tensor.h"
#include "Parameter.h"
#include "../test.h"
#include <iostream>
#include <stdexcept>
#include <vector>

#define N 8192  // 输入向量长度
#define D 8192   // 输出向量长度

void check_add(){
    Title("check_add");
    DeviceManager& manager = DeviceManager::getInstance();

    Device * cpu = manager.getDevice("cpu");
    Device * cuda = manager.getDevice("cuda");

    int batch_size = 5;
    // 分配主机内存
    Tensor<float> x1_cpu(batch_size, N, "cpu", "x1_cpu");
    Tensor<float> x2_cpu(batch_size, N, "cpu", "x2_cpu");

    rand_init(x1_cpu, N * batch_size);
    rand_init(x2_cpu, N * batch_size);

    Tensor<float> x1_cuda(x1_cpu);
    Tensor<float> x2_cuda(x2_cpu);

    x1_cuda.to("cuda");
    x2_cuda.to("cuda");

    cpu->F->add(x2_cpu, x1_cpu, x2_cpu, N, batch_size);
    cuda->F->add(x2_cuda, x1_cuda, x2_cuda, N, batch_size);

    x2_cuda.to("cpu");

    // 比较结果
    check(x2_cuda, x2_cpu, N * batch_size, "add");
}

void check_parameter() {
    Title("check_parameter");
    size_t len = 10;
    Parameter<int> param1(1, 10, "cpu", "param");
    Parameter<int> param2(1, 10, "cpu", "param");

    for(int i = 0; i < len; i++) {
        param1[i] = i;
        param2[i] = i + 10;
    }

    param1.setShared();
    param1.copy(3, param2, 2, 4);

    
    for(int i = 0; i < len; i++)
        std::cout << param1[i] << " ";
    std::cout << std::endl;

    for(int i = 0; i < len; i++)
        std::cout << param2[i] << " ";
    std::cout << std::endl;

    std::cout << param1 << " " << param1.Name() << std::endl;
    param1.to("cuda");
    std::cout << param1 << " " << param1.Name() << std::endl;
    std::cout << param1.Device() << std::endl;

    try {
        param1.copy(3, param2, 2, 4);
    } catch (const std::logic_error& e) {
        // 异常处理代码
        std::cout << "Caught exception: " << e.what() << std::endl;
    }
}

int main() {

    check_parameter();
    check_add();

    return 0;
}
