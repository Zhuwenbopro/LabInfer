# 编译器
CXX = g++

# 路径设置
SRC_DIR = ../../src
DEVICE_DIR = $(SRC_DIR)/device
TEST_DIR = .

# 目标文件
TARGET = $(TEST_DIR)/device_test

# 源文件
TEST_SRC = $(TEST_DIR)/main.cc
DEVICE_SRC = $(DEVICE_DIR)/Device.cc

# 编译标志
CXXFLAGS = -std=c++11 -I$(DEVICE_DIR)

# 目标规则
all: $(TARGET)

# 编译规则，生成测试可执行文件
$(TARGET): $(TEST_SRC) $(DEVICE_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_SRC) $(DEVICE_SRC)

# 清理
clean:
	rm -f $(TARGET)

.PHONY: all clean

