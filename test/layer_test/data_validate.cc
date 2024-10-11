#include <iostream>
#include <fstream>
#include <memory>

int main() {
    // 定义张量的形状
    const size_t dim0 = 128256;
    const size_t dim1 = 2048;
    const size_t total_elements = dim0 * dim1;

    // 分配主机内存
    auto tensor_data = std::make_unique<float[]>(total_elements);

    // 打开二进制文件
    std::ifstream infile("model_embed_tokens_weight.bin", std::ios::binary);
    if (!infile) {
        std::cerr << "无法打开文件 model_embed_tokens_weight.bin" << std::endl;
        return 1;
    }

    // 检查文件大小
    infile.seekg(0, std::ios::end);
    std::streamsize file_size = infile.tellg();
    infile.seekg(0, std::ios::beg);

    if (file_size != static_cast<std::streamsize>(total_elements * sizeof(float))) {
        std::cerr << "文件大小与预期不匹配" << std::endl;
        return 1;
    }

    // 读取数据
    infile.read(reinterpret_cast<char*>(tensor_data.get()), total_elements * sizeof(float));

    if (!infile) {
        std::cerr << "读取文件错误，仅读取了 " << infile.gcount() << " 字节" << std::endl;
        return 1;
    }

    infile.close();

/******************************** 验证 embedding运算 猜想正确 *********************************/

    int input[] = {128000,    791,   1401,    311,   2324,    374};

    float* embedding_tensor = new float[6 * dim1];

    // 打开二进制文件
    std::ifstream ifs("embedding_tensor.bin", std::ios::binary);
    if (!ifs) {
        std::cerr << "无法打开文件 embedding_tensor.bin" << std::endl;
        return 1;
    }

    // 检查文件大小
    ifs.seekg(0, std::ios::end);
    file_size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    if (file_size != static_cast<std::streamsize>(6 * dim1 * sizeof(float))) {
        std::cerr << "文件大小与预期不匹配" << std::endl;
        return 1;
    }

    // 读取数据
    ifs.read((char*)embedding_tensor, 6 * dim1 * sizeof(float));

    if (!ifs) {
        std::cerr << "读取文件错误，仅读取了 " << ifs.gcount() << " 字节" << std::endl;
        return 1;
    }

    ifs.close();

    for(int i = 0; i < 6; i++){
        int token_id = input[i];
        for(int j = 0; j < dim1; j++){
            if(embedding_tensor[i*dim1+j] != tensor_data.get()[token_id*dim1 + j]){
                std::cout << "i = " << i << " j = " << j << std::endl;
                std::cout << "don't match " << embedding_tensor[i*dim1+j] << " vs " << tensor_data.get()[token_id*dim1 + j] << std::endl;
                break;
            }
        }
    }

    std::cout << "matched !! congratulations" << std::endl;
    return 0;
}
