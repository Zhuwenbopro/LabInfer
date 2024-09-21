# 编译器
CXX = g++
NVCC = nvcc

# 路径设置
SRC_DIR = ../../src
CUDA_DIR = $(SRC_DIR)/function/cuda
TEST_DIR = .
BUILD_DIR = .

# CUDA 和编译选项
CUDA_LIB_DIR = /usr/local/cuda/lib64/
CXXFLAGS = -std=c++11 -I$(SRC_DIR)/function/cuda  # 添加头文件路径
LDFLAGS = -L$(BUILD_DIR) -Wl,-rpath=$(BUILD_DIR) -L$(CUDA_LIB_DIR) -lcudart

# 主程序源文件
MAIN_SRC = $(TEST_DIR)/main.cc
MAIN_TARGET = $(TEST_DIR)/test

# CUDA 源文件和对应的动态库
CUDA_SRCS = $(wildcard $(CUDA_DIR)/*.cu)
CUDA_BASENAMES = $(basename $(notdir $(CUDA_SRCS)))
CUDA_LIBS = $(addprefix $(BUILD_DIR)/lib, $(addsuffix .so, $(CUDA_BASENAMES)))

# 获取库名称（去掉 'lib' 前缀和 '.so' 后缀）
LIB_NAMES = $(patsubst lib%.so, %, $(notdir $(CUDA_LIBS)))

# 检查是否为调试模式
ifeq ($(DEBUG), 1)
    NVCCFLAGS = -DDEBUG -G -g -Xcompiler -fPIC -shared
else
    NVCCFLAGS = -Xcompiler -fPIC -shared
endif

# 默认目标
all: $(MAIN_TARGET)

# 编译主程序
$(MAIN_TARGET): $(MAIN_SRC) $(CUDA_LIBS)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(addprefix -l, $(LIB_NAMES))

# 编译 CUDA 源文件为动态库
$(BUILD_DIR)/lib%.so: $(CUDA_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# 清理
clean:
	rm -f $(MAIN_TARGET) $(CUDA_LIBS)
