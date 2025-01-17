# =============================================================================
# 编译器设置
# =============================================================================
CXX := g++
NVCC := nvcc

# =============================================================================
# 路径设置
# =============================================================================
SRC_DIR := ../../src
DEVICE_DIR := $(SRC_DIR)/device
BUILD_DIR := ../../build
THIRD_PARTY_DIR := ./../../third_party

INCLUDES := -I./../../include -I$(DEVICE_DIR) -I$(THIRD_PARTY_DIR)

# =============================================================================
# 查找所有 .cc 和 .cu 文件
# =============================================================================
CPU_SRC := ../../src/device/function/cpu/CPUFunction.cc ../../src/layer/layer.cc
CPU_OBJ := $(addprefix $(BUILD_DIR)/, $(notdir $(CPU_SRC:.cc=.o)))

CUDA_SRC := $(shell find $(SRC_DIR) -type f -name '*.cu')
CUDA_OBJ := $(addprefix $(BUILD_DIR)/, $(notdir $(CUDA_SRC:.cu=.o)))

# 主程序源文件
MAIN_SRC := main.cc
MAIN_OBJ := $(BUILD_DIR)/main.o

# =============================================================================
# 编译标志
# =============================================================================
MACRO := -DUSE_CUDA
CXXSTAND := -std=c++17
CXXFLAGS := $(CXXSTAND) $(INCLUDES) $(MACRO) -O3 -fopenmp  # 添加 -fopenmp
NVCCFLAGS := $(CXXSTAND) $(INCLUDES) $(MACRO) -O3 -Xcompiler -fopenmp # 使用 -Xcompiler 传递给 g++

# =============================================================================
# 最终目标
# =============================================================================
TARGET := test

# =============================================================================
# 默认目标
# =============================================================================
all: $(TARGET)

# =============================================================================
# 链接目标
# =============================================================================
LDFLAGS := -lopenblas -lcublas

$(TARGET): $(MAIN_OBJ) $(CPU_OBJ) $(CUDA_OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)
# =============================================================================
# 编译规则
# =============================================================================

# 确保 build 目录存在
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# 编译 main.cc 为 main.o
$(MAIN_OBJ): $(MAIN_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

# 定义编译 .cc 文件为 .o 文件的规则
define compile_cpu
$(BUILD_DIR)/$(notdir $(1:.cc=.o)): $(1) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -MMD -c $(1) -o $$@
endef

# 定义编译 .cu 文件为 .o 文件的规则
define compile_cuda
$(BUILD_DIR)/$(notdir $(1:.cu=.o)): $(1) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -MMD -c $(1) -o $$@
endef

# 为每个 .cc 文件生成编译规则
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
#	mv ./*.bin ~/data

#data:
#	mv ~/data/k.bin ./
#	mv ~/data/q.bin ./
#	mv ~/data/v.bin ./
#	mv ~/data/o.bin ./

# =============================================================================
# 声明伪目标
# =============================================================================
.PHONY: all clean
