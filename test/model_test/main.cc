#include "models/llama.h"
#include <chrono>

// ANSI color codes
#define RESET   "\033[0m"
#define RED     "\033[31m"      // Red
#define GREEN   "\033[32m"      // Green

#define N 4096  // 输入向量长度
#define D 4096   // 输出向量长度

// void check_pass(const std::string& message);
// void check_error(const std::string& message);
// bool compare_results(const float *a, const float *b, int size, float tolerance = 1e-3);

// void read_bin(float* ptr, size_t num, const std::string& filename);

// void check_lm_head() {
//     size_t vocal_size = 128256;
//     size_t hidden_size = 2048;

//     std::vector<std::vector<size_t>> input_ids = {{128000, 791, 1401, 311, 2324, 374}};
//     std::vector<std::vector<size_t>> position = {{0, 1, 2, 3, 4, 5}};
//     std::vector<size_t> uid = {112358};

//     Linear linear(hidden_size, vocal_size, "model.embed_tokens");
//     linear.load_state("./model.safetensors", true);

//     std::shared_ptr<float []> ptr(new float[6*hidden_size]);
//     read_bin(ptr.get(), 6 * hidden_size, "norm.bin");

//     Tensor x(6, hidden_size, "cpu", uid, {6});
//     x.addPos(position);
//     x.setUid(uid);
//     x.setValue(ptr);

//     x = linear.forward(x);
//     x = x.tail();

//     float* p = new float[x.elemLen()];
//     read_bin(p, x.elemLen(), "lm_head.bin");

//     if(compare_results(p, x, x.elemLen(), 5e-2)) {
//         check_pass("lm_head check pass");
//     } else {
//         check_error("lm_head check not pass");
//     }

//     float* logist = new float[x.elemLen()];
//     read_bin(logist, x.elemLen(), "logits.bin");

//     if(compare_results(p, logist, x.elemLen(), 5e-3)) {
//         check_pass("logist check pass");
//     } else {
//         check_error("logist check not pass");
//     }
// }

int main() {

    // check_lm_head();
    // return 1;

    Config config("config.json");
    Llama model(config);

    std::cout << "loading model..." << std::endl;
    model.load_state("./model.safetensors", true);
    std::cout << "loaded model" << std::endl;

    std::vector<std::vector<size_t>> input_ids = {{128000, 791, 1401, 311, 2324, 374}};
    std::vector<std::vector<size_t>> position = {{0, 1, 2, 3, 4, 5}};
    std::vector<size_t> uid = {112358};

    Tensor x(input_ids);
    x.addPos(position);
    x.setUid(uid);

    // x.to("cuda");
    // model.to("cuda");
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 10; i++) {
        //std::cout << i << std::endl;
        x = model.forward(x);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << " executed in " << elapsed.count() << " ms.\n";

    // return -1;
    
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

