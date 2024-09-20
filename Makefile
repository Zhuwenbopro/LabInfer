# 编译器
CXX = g++
NVCC = nvcc

# 路径设置
SRC_DIR = src
BUILD_DIR = build
CUDA_DIR = $(SRC_DIR)/function/cuda
MAIN_DIR = $(SRC_DIR)/main

# CUDA 和编译选项
CUDA_LIB_DIR = /usr/local/cuda-12.1/targets/x86_64-linux/lib
CXXFLAGS = -std=c++11
LDFLAGS = -L$(BUILD_DIR) -Wl,-rpath=$(BUILD_DIR) -L$(CUDA_LIB_DIR) -lcudart
NVCCFLAGS = -Xcompiler -fPIC  # 添加 -fPIC 选项

# CUDA 源文件
CUDA_SRCS = $(wildcard $(CUDA_DIR)/*.cu)
CUDA_OBJS = $(patsubst $(CUDA_DIR)/%.cu, $(BUILD_DIR)/%.o, $(CUDA_SRCS))
CUDA_LIBS = $(patsubst $(CUDA_DIR)/%.cu, $(BUILD_DIR)/lib%.so, $(CUDA_SRCS))

# 主程序源文件
MAIN_SRC = $(MAIN_DIR)/main.cc
MAIN_TARGET = $(BUILD_DIR)/main

# 默认目标
all: $(MAIN_TARGET)

# 编译 CUDA 源文件为动态库
$(BUILD_DIR)/lib%.so: $(BUILD_DIR)/%.o
	$(NVCC) -shared -o $@ $<

# 编译 CUDA 源文件为 .o 文件
$(BUILD_DIR)/%.o: $(CUDA_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# 编译主程序，并链接所有 CUDA 动态库
$(MAIN_TARGET): $(MAIN_SRC) $(CUDA_LIBS)
	$(CXX) -o $@ $(MAIN_SRC) $(LDFLAGS) -lmykernel

# 清理生成的文件
clean:
	rm -f $(BUILD_DIR)/*.o $(BUILD_DIR)/lib*.so $(MAIN_TARGET)

.PHONY: all clean

