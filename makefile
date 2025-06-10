# InferServer Makefile

# 编译器和编译选项
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -I include
LDFLAGS = -static -lpthread -static-libgcc -static-libstdc++

# 源文件和目标文件
SRCS = src/main.cpp src/Worker.cpp src/Engine.cpp 
OBJS = $(SRCS:.cpp=.o)
TARGET = infer_server

# 默认目标
all: $(TARGET)

# 链接目标文件生成可执行文件
$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

# 编译规则
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 清理目标
clean:
	del /Q $(subst /,\,$(OBJS)) $(TARGET).exe

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

.PHONY: all clean rebuild run help