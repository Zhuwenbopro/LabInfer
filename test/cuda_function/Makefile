# 编译器
CXX := g++
NVCC := nvcc

# 路径设置
SRC_DIR := ../../src
CUDA_DIR := $(SRC_DIR)/function/cuda
BUILD_DIR := .

# 包含目录（如果有）
INCLUDES := -I./$(CUDA_DIR)

# CUDA 编译参数
# 假设 CUDA 源文件扩展名为 .cu
CUDA_SRC := $(wildcard $(CUDA_DIR)/*.cu)
CUDA_OBJ := $(patsubst $(CUDA_DIR)/%.cu, $(BUILD_DIR)/%.o, $(CUDA_SRC))
CUDA_LIB := libCudaFunctionLibrary.so
CUDA_LIB_DIR = /usr/local/cuda/lib64/
LDFLAGS = -L$(BUILD_DIR) -Wl,-rpath=$(BUILD_DIR) -L$(CUDA_LIB_DIR) -lcudart

# 主程序源文件
MAIN_SRC = main.cc
MAIN_TARGET = test

# 默认目标
all: $(MAIN_TARGET)

# 链接生成动态库
$(BUILD_DIR)/$(CUDA_LIB): $(CUDA_OBJ)
	$(NVCC) -shared -o $@ $^ -lcudart

# 规则：编译 .cu 文件
$(BUILD_DIR)/%.o: $(CUDA_DIR)/%.cu
	$(NVCC) -c -Xcompiler -fPIC $(INCLUDES) $< -o $@

# 规则：编译并链接 main.cc 成可执行文件 main
$(MAIN_TARGET): $(MAIN_SRC) $(BUILD_DIR)/$(CUDA_LIB) 
	$(CXX) $(INCLUDES) main.cc -L$(BUILD_DIR) -lCudaFunctionLibrary $(LDFLAGS) -o $(MAIN_TARGET)

# 清理生成文件
clean:
	rm -f $(BUILD_DIR)/*.o $(BUILD_DIR)/$(CUDA_LIB) $(MAIN_TARGET)