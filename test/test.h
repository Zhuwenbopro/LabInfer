#ifndef TEST_DEBUG
#define TEST_DEBUG

#include <iostream>
#include <cstdlib>  // 用于rand函数
#include <ctime>    // 用于时间种子
#include <random>
#include <fstream>


// ANSI color codes
#define RESET   "\033[0m"
#define RED     "\033[31m"      // Red
#define GREEN   "\033[32m"      // Green
#define die(...) do{printf(__VA_ARGS__); fputc('\n',stdout); exit(EXIT_FAILURE);}while(0);

#include <iomanip> // 用于设置输出格式

// 定义控制台宽度，默认设为80列
const int CONSOLE_WIDTH = 80;

// Title 函数定义
void Title(const std::string &title) {
    // 计算装饰线的长度（与控制台宽度相同）
    std::string decoration(CONSOLE_WIDTH, '=');
    // 计算标题左侧的填充空格数以实现居中
    int padding = (CONSOLE_WIDTH - title.length()) / 2;
    if (padding < 0) padding = 0; // 防止负数
    // 打印装饰线
    std::cout << decoration << std::endl;
    // 打印居中的标题
    std::cout << std::setw(padding) << "" << title << std::endl;
    // 打印装饰线
    std::cout << decoration << std::endl;
}


void write_bin(const std::string& filename, float* ptr, size_t size) {
    std::ofstream outFile(filename, std::ios::binary);

    if (!outFile) {
        std::cerr << "无法打开文件 " + filename << std::endl;
        return ;
    }

    // 写入数组数据到文件
    outFile.write(reinterpret_cast<char*>(ptr), size * sizeof(float));
    
    // 关闭文件
    outFile.close();

    std::cout << "数据已存储到文件 " + filename << std::endl;
}

void read_bin(const std::string& filename, float* ptr, size_t size) {
    std::ifstream inFile(filename, std::ios::binary);

    if (!inFile) {
        std::cerr << "无法打开文件" << std::endl;
        return ;
    }
    // 获取文件大小
    inFile.seekg(0, std::ios::end);  // 移动到文件末尾
    std::streampos fileSize = inFile.tellg();  // 获取文件大小
    inFile.seekg(0, std::ios::beg);  // 回到文件开始
    if(fileSize / sizeof(float) != size) {
        std::cerr << "文件尺寸对不上" << std::endl;
        return ;
    }
    inFile.read(reinterpret_cast<char*>(ptr), fileSize);

    inFile.close();
}

void check_pass(const char* message);
void check_error(const char* message);
bool compare_results(const float *a, const float *b, int size, float tolerance= 1e-3f);
void rand_init(float* ptr, int size);

void check(const float *a, const float *b, int size, const std::string& item, float tolerance=1e-3f) {
    if (compare_results(a, b, size)) {
        check_pass(("[" + item + "] CUDA and CPU results match.").c_str());
    } else {
        check_error(("[" + item + "] CUDA and CPU results do not match!").c_str());
    }

    for(int i = 0; i < 5; i++) {
        if(i >= size) break;
        std::cout << a[i] << " vs " << b[i] << std::endl;
    }
}


void check_pass(const char*  message){
    std::cout << GREEN << message << RESET << std::endl;
}

void check_error(const char*  message){
    std::cout << RED << message << RESET << std::endl;
}

bool compare_results(const float *a, const float *b, int size, float tolerance) {
    for (int i = 0; i < size; ++i) {
        if (std::fabs(a[i] - b[i]) > tolerance) {
            std::cout << "Difference at index " << i << ": " << a[i] << " vs " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

// 随机数生成器
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

void rand_init(float* ptr, int size){
    for (int i = 0; i < size; ++i) {
        ptr[i] = dist(gen);
    }    
}

#endif