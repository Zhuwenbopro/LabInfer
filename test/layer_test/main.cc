#include "layers.h"
#include "models.h"
#include "Tensor.h"
#include <fstream>
#include <memory>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>  // 用于rand函数
#include <ctime>    // 用于时间种子


// ANSI color codes
#define RESET   "\033[0m"
#define RED     "\033[31m"      // Red
#define GREEN   "\033[32m"      // Green

#define N 4096  // 输入向量长度
#define D 4096   // 输出向量长度

std::unordered_map<std::string, std::shared_ptr<float []>> states;

void check_tensor(Tensor& tensor);

void check_pass(const std::string& message);
void check_error(const std::string& message);
bool compare_results(const float *a, const float *b, int size, float tolerance = 1e-3);

void read_bin(float* ptr, size_t num, const std::string& filename);

int main() {

    Config config("config.json");

    const size_t hidden_state = 2048;
    Llama llama = Llama(config);
    std::cout << "loading model..." << std::endl;
    llama.load_state("./model.safetensors", true);
    std::cout << "loaded model" << std::endl;


    std::vector<std::vector<size_t>> input_ids = {{128000, 791, 1401, 311, 2324, 374}};
    Tensor x({input_ids});
    x.setName("input_ids");
    // check_tensor(x);

    Tensor y(x, hidden_state);
    Tensor y_check = y.copy();
    // read_bin(y_check, y_check.Size(), "embedding_tensor.bin");
    read_bin(y_check, y_check.Size(), "norm.bin");


    llama.forward(y, x);


    if (compare_results(y_check, y, y_check.Size(), 6e-2)) {
        check_pass("[" + llama.Name() +"] " + llama.Device() + " results correct.");
    } else {
        check_error("[" + llama.Name() +"] " + llama.Device() + " results error!");
    }
    
    std::cout << y_check[3000] << "end" << y[3000] << std::endl;

    return 0;
}

void check_tensor(Tensor& tensor) {

    std::cout << std::endl << " check tensor:" << std::endl;
    for(int i = 0; i < tensor.Size(); i++) {
        std::cout << tensor[i] << std::endl;
    }

    std::cout << "Name:" << tensor.Name() << std::endl;
    std::cout << "Device:" << tensor.Device() << std::endl;
    std::cout << "elemLen:" << tensor.elemLen() << std::endl;
    std::cout << "elemNum:" << tensor.elemNum() << std::endl;

    std::cout << "batchSize:" << tensor.batchSize() << std::endl;

    std::cout << "Uid:" << std::endl;
    const std::vector<size_t>& uid = tensor.Uid();
    for(int i = 0; i < uid.size(); i++) {
        std::cout << uid[i] << std::endl;
    }

    std::cout << "Seq:" << std::endl;
    const std::vector<size_t>& seq = tensor.Seq();
    for(int i = 0; i < seq.size(); i++) {
        std::cout << seq[i] << std::endl;
    }

    std::cout << "check tensor end" << std::endl;
}

void check_pass(const std::string&  message){
    std::cout << GREEN << message << RESET << std::endl;
}

void check_error(const std::string&  message){
    std::cout << RED << message << RESET << std::endl;
}

float fabs(float c){
    return c >= 0 ?  c : -c;
}

bool compare_results(const float *a, const float *b, int size, float tolerance) {
    std::cout << "Comparing results...   size:" << size << std::endl;
    bool flag = true;
    for (int i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > tolerance) {
            std::cout << "Difference at index " << i << ": " << a[i] << " vs " << b[i] << std::endl;
            flag = false;
            break;
        }
    }
    return flag;
}


void read_bin(float* ptr, size_t num, const std::string& filename) {
    // 打开二进制文件
    std::ifstream weight_file(filename, std::ios::binary);
    if (!weight_file) {
        std::cerr << "无法打开文件 " << filename << std::endl;
        return;
    }

    // 检查文件大小
    weight_file.seekg(0, std::ios::end);
    std::streamsize file_size = weight_file.tellg();
    weight_file.seekg(0, std::ios::beg);

    if (file_size != static_cast<std::streamsize>(num * sizeof(float))) {
        std::cerr << "文件大小与预期不匹配" << std::endl;
        return;
    }

    // 读取数据
    weight_file.read((char*)ptr, num * sizeof(float));

    if (!weight_file) {
        std::cerr << "读取文件错误，仅读取了 " << weight_file.gcount() << " 字节" << std::endl;
        return;
    }

    weight_file.close();
}
