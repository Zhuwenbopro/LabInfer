#include "Manager.h"
#include "Function.h"
#include "Tensor.h"


int main() {
    Manager& manager = Manager::getInstance();

    Function& F = manager.getFunction("cpu");

    Device& CPU = manager.getDevice("cpu");

    size_t size = 2;
    float* x = CPU.allocate(size);
    float* M = CPU.allocate(size);
    float* x_out = CPU.allocate(size);

    x[0] = 1; x[1] = 2;
    M[0] = 1; M[1] = 2; M[2] = 3; M[3] = 4; 

    F.matmul(x_out, x, M, size, size);

    std::cout << x_out[0] << "  " << x_out[1] << std::endl;
    
    return 0;
}