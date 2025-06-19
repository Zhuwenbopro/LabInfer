# =============================================================================
# 编译器设置
# =============================================================================
CXX := g++
NVCC := nvcc

# =============================================================================
# 路径设置
# =============================================================================
SRC_DIR := ../../../src
BUILD_DIR := ./build
INCLUDES := -I./../../../include -I./../../../thirdparty

# =============================================================================
# 查找所有 .cpp 和 .cu 文件
# =============================================================================
# C++ 源文件
CPU_SRC := $(SRC_DIR)/CPU/CPUMemoryManager.cpp \
		   $(SRC_DIR)/ops/linear/linear_cpu.cpp \
		   $(SRC_DIR)/ops/rope/rope_cpu.cpp \
		   $(SRC_DIR)/ops/softmax/softmax_cpu.cpp \
		   $(SRC_DIR)/ops/silu/silu_cpu.cpp \
		   $(SRC_DIR)/ops/add_elem/add_elem_cpu.cpp \
		   $(SRC_DIR)/ops/max_index/max_index_cpu.cpp \
		   $(SRC_DIR)/ops/rms_norm/rms_norm_cpu.cpp \
		   $(SRC_DIR)/ops/multiply_elem/multiply_elem_cpu.cpp 
CPU_OBJ := $(addprefix $(BUILD_DIR)/, $(notdir $(CPU_SRC:.cpp=.o)))

# CUDA 源文件
CUDA_SRC := $(SRC_DIR)/CUDA/CUDAMemoryManager.cu \
		    $(SRC_DIR)/ops/linear/linear_cuda.cu \
			$(SRC_DIR)/ops/rope/rope_cuda.cu \
			$(SRC_DIR)/ops/softmax/softmax_cuda.cu \
			$(SRC_DIR)/ops/silu/silu_cuda.cu \
			$(SRC_DIR)/ops/add_elem/add_elem_cuda.cu \
			$(SRC_DIR)/ops/max_index/max_index_cuda.cu \
			$(SRC_DIR)/ops/rms_norm/rms_norm_cuda.cu \
			$(SRC_DIR)/ops/multiply_elem/multiply_elem_cuda.cu
CUDA_OBJ := $(addprefix $(BUILD_DIR)/, $(notdir $(CUDA_SRC:.cu=.o)))

# 主程序源文件
MAIN_SRC := main.cu
MAIN_OBJ := $(BUILD_DIR)/main.o

# =============================================================================
# 编译标志
# =============================================================================
MACRO := -DUSE_CUDA
CXXSTAND := -std=c++17
# 给 C++ 编译器的标志
CXXFLAGS := $(CXXSTAND) $(INCLUDES) $(MACRO) -O3 -fopenmp
# 给 NVCC 编译器的标志, -fopenmp 需要通过 -Xcompiler 传递
NVCCFLAGS := $(CXXSTAND) $(INCLUDES) $(MACRO) -O3 -Xcompiler -fopenmp

# =============================================================================
# 最终目标
# =============================================================================
TARGET := test
all: $(TARGET)

# =============================================================================
# 链接目标
# =============================================================================
LDFLAGS := -Xcompiler -fopenmp -lopenblas

$(TARGET): $(MAIN_OBJ) $(CPU_OBJ) $(CUDA_OBJ)
	$(NVCC) -o $@ $^ $(LDFLAGS)
# =============================================================================
# 编译规则
# =============================================================================

# 确保 build 目录存在
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# 编译 main.cu 为 main.o
$(MAIN_OBJ): $(MAIN_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -MMD -c $< -o $@

# 定义编译 .cpp 文件为 .o 文件的规则
define compile_cpu
$(BUILD_DIR)/$(notdir $(1:.cpp=.o)): $(1) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -MMD -c $(1) -o $$@
endef

# 定义编译 .cu 文件为 .o 文件的规则
define compile_cuda
$(BUILD_DIR)/$(notdir $(1:.cu=.o)): $(1) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -MMD -c $(1) -o $$@
endef

# 为每个 .cpp 文件生成编译规则
$(foreach src,$(CPU_SRC),$(eval $(call compile_cpu,$(src))))

# 为每个 .cu 文件生成编译规则
$(foreach src,$(CUDA_SRC),$(eval $(call compile_cuda,$(src))))

# 包含依赖文件
-include $(CPU_OBJ:.o=.d) $(CUDA_OBJ:.o=.d) $(MAIN_OBJ:.o=.d)

# =============================================================================
# 清理目标
# =============================================================================
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

# =============================================================================
# 声明伪目标
# =============================================================================
.PHONY: all clean