// CudaFunctionLibrary.h
#ifndef CUDA_FUNCTION_H
#define CUDA_FUNCTION_H

// 包含所有相关的头文件
#include "Function.h"
#include "matmul.h"
#include "rmsnorm.h"
// 如果有更多的头文件，继续添加
// #include "another_function.h"

class CUDAFunction : public Function {
public:
    CUDAFunction() {}
}

#endif // CUDA_FUNCTION_H
