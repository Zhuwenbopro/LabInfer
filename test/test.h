#include <iostream>
#include <cstdlib>  // 用于rand函数
#include <ctime>    // 用于时间种子

// ANSI color codes
#define RESET   "\033[0m"
#define RED     "\033[31m"      // Red
#define GREEN   "\033[32m"      // Green

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

float fabs(float c){
    return c >= 0 ?  c : -c;
}

bool compare_results(const float *a, const float *b, int size, float tolerance) {
    for (int i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > tolerance) {
            std::cout << "Difference at index " << i << ": " << a[i] << " vs " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

void rand_init(float* ptr, int size){
    // 设置随机数种子
    std::srand(static_cast<unsigned int>(std::time(0)));

    for (int i = 0; i < size; ++i) {
        ptr[i] = static_cast<float>(rand()) / RAND_MAX;
    }    
}
