# 编译器
CXX = g++

# 路径设置
SRC_DIR = ../../src
DEVICE_DIR = $(SRC_DIR)/variable

# 编译标志
CXXFLAGS = -std=c++11 -I$(DEVICE_DIR)

# 目标规则
all: test

# 编译规则，生成测试可执行文件
test: main.cc
	$(CXX) $(CXXFLAGS) -o $@ main.cc

# 清理
clean:
	rm -f test

.PHONY: all clean

