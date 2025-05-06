# 定义编译器和目标名称
NVCC        = nvcc
CXX         = g++
TARGET      = my_program                # 最终生成的可执行文件名称

# CUDA 源文件：注意这里指向的是 ../../src/device/CUDA 目录下的文件
CUDA_SRCS   = ../../../src/device/CUDA/CUDAFunction.cu \
              ../../../src/device/CUDA/CUDAAllocator.cu \
              ../../../src/device/CUDA/CUDA.cu

# 主机端源文件（例如测试入口 main.cc 位于当前目录）
HOST_SRCS   = main.cc

# 指定中间文件存放目录（相对于当前目录）
BUILD_DIR   = build

# 自动生成对象文件列表（仅保留文件名，以避免由于目录信息导致的匹配问题）
CUDA_OBJS   = $(addprefix $(BUILD_DIR)/, $(notdir $(CUDA_SRCS:.cu=.o)))
HOST_OBJS   = $(addprefix $(BUILD_DIR)/, $(notdir $(HOST_SRCS:.cc=.o)))
OBJS        = $(CUDA_OBJS) $(HOST_OBJS)

# 设置 VPATH，在当前目录和 CUDA 源码所在目录查找依赖文件
VPATH = . ../../../src/device/CUDA

# 包含路径，需要调整到从当前目录访问项目根目录下的 include 及 CUDA 目录
INCLUDES    := -I../../../include -I../../../src/device/CUDA
CXXSTAND    := -std=c++17
MACRO       =
CXXFLAGS    := $(CXXSTAND) $(INCLUDES) $(MACRO) -O3 -fopenmp
NVCCFLAGS   := $(CXXSTAND) $(INCLUDES) $(MACRO) -O3 -Xcompiler -fopenmp
LDFLAGS     := -lopenblas -lcublas

# 默认目标：编译生成可执行文件
all: $(TARGET)

# 链接所有对象文件生成最终可执行文件，使用 NVCC 链接（NVCC 能链接 CUDA 代码和常规 C++ 对象）
$(TARGET): $(OBJS)
	$(NVCC) -o $(TARGET) $(OBJS) $(LDFLAGS)

# 编译 CUDA 源文件（.cu文件），目标为 BUILD_DIR 下的 .o 文件
$(BUILD_DIR)/%.o: %.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# 编译 C++ 源文件（.cc文件），目标为 BUILD_DIR 下的 .o 文件
$(BUILD_DIR)/%.o: %.cc
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 清理生成的中间文件和最终可执行文件
.PHONY: clean
clean:
	rm -f $(TARGET) $(OBJS)
