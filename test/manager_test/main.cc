#include "Manager.h"
#include "Function.h"
#include "Tensor.h"


int main() {
    Manager& manager = Manager::getInstance();

    Function& F = manager.getFunction("cpu");

    size_t size = 2;
    std::shared_ptr<float[]> x = manager.allocate(size, "cpu");
    std::shared_ptr<float[]> M = manager.allocate(size*size, "cpu");
    std::shared_ptr<float[]> y = manager.allocate(size, "cpu");

    x[0] = 1; x[1] = 2;
    M[0] = 1; M[1] = 2; M[2] = 3; M[3] = 4; 

    F.matmul(y.get(), x.get(), M.get(), size, size);

    std::cout << "x:\t" << x[0] << "  " << x[1] << std::endl << std::endl;
    std::cout << "M:\t"<< M[0] << "  " << M[2] << std::endl
                 << "\t"<< M[1] << "  " << M[3] << std::endl << std::endl;
    std::cout << "y=xM:\t" << y[0] << "  " << y[1] << std::endl << std::endl;
    
    return 0;
}