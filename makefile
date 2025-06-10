# InferServer Makefile

# 编译器和编译选项
NVCC = nvcc
# 使用 nvcc 作为编译器，它能同时处理 .cpp 和 .cu 文件
# CXXFLAGS 适用于所有编译
CXXFLAGS = -std=c++17 -I include
# LDFLAGS 是链接时用的库
LDFLAGS = -lpthread

# 目录和最终目标
BUILD_DIR = build
TARGET = infer_server

# --- 源文件和目标文件 ---
# 自动查找所有 .cpp 和 .cu 源文件
CPP_SRCS   = $(wildcard src/*.cpp)
CU_SRCS    = $(wildcard src/CUDA/*.cu)

# 根据源文件自动生成目标 (.o) 文件列表
# patsubst (pattern substitution) 是更灵活的替换方式
# 将 src/%.cpp -> build/%.o
# 将 src/CUDA/%.cu -> build/%.o
OBJS       = $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SRCS)) \
             $(patsubst src/CUDA/%.cu,$(BUILD_DIR)/%.o,$(CU_SRCS))

# --- 编译和链接规则 ---

# 默认目标
all: $(TARGET)

# 链接目标文件生成可执行文件
# $@ 代表目标文件 (infer_server)
# $^ 代表所有依赖项 (所有的 .o 文件)
$(TARGET): $(OBJS)
	@echo "Linking..."
	$(NVCC) -o $@ $^ $(LDFLAGS)
	@echo "Build finished: $(TARGET)"

# 确保在编译任何东西之前，构建目录已存在
# 这是一个 order-only 依赖，意味着 build 目录必须在编译 .o 文件前存在
# 但如果 build 目录的修改时间比 .o 文件新，不会触发重新编译 .o 文件
# 我们将它作为每个 .o 规则的依赖来简化
$(BUILD_DIR):
	@echo "Creating directory: $(BUILD_DIR)"
	@mkdir -p $(BUILD_DIR)

# 为 .cpp 文件定义编译规则
# $< 代表第一个依赖项 (源文件)
# | $(BUILD_DIR) 确保构建目录先被创建
$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	@echo "Compiling C++: $< -> $@"
	$(NVCC) $(CXXFLAGS) -c $< -o $@

# 为 .cu 文件定义编译规则
$(BUILD_DIR)/%.o: src/CUDA/%.cu | $(BUILD_DIR)
	@echo "Compiling CUDA: $< -> $@"
	$(NVCC) $(CXXFLAGS) -c $< -o $@

# --- 其他实用规则 ---

# 清理目标
clean:
	@echo "Cleaning up..."
	@rm -rf $(BUILD_DIR) $(TARGET)

# 重新编译
rebuild: clean all

# 运行程序
run: $(TARGET)
	./$(TARGET)

# 帮助信息
help:
	@echo "使用方法:"
	@echo "  make       - 编译项目"
	@echo "  make clean - 清理编译生成的文件"
	@echo "  make rebuild - 重新编译整个项目"
	@echo "  make run   - 运行编译后的程序"

# 声明这些目标是“伪目标”，它们不代表真实的文件
.PHONY: all clean rebuild run help