//#include "layers.h"
#include "layers/layers.h"
//#include "models.h"
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

void check_pass(const std::string& message);
void check_error(const std::string& message);
bool compare_results(const float *a, const float *b, int size, float tolerance = 1e-3);

void read_bin(float* ptr, size_t num, const std::string& filename);

void check_linear() {
    std::cout << " check linear " << std::endl << std::endl;
    size_t size_in = 2;
    size_t size_out = 2;
    std::unordered_map<std::string, std::shared_ptr<float[]>> weights;
    weights["weight"] = std::shared_ptr<float[]>(new float[4]{1.0f, 2.0f, 3.0f, 4.0f});

    Linear linear(size_in, size_out);
    linear.load_weights(weights);

    Tensor x(1, 2, "cpu", {1}, {2});
    x[0] = 1; x[1] = 2;
    Tensor y = linear.forward(x);
    // x [1, 2]
    // w [[1,2],[3,4]] 列主导
    // y [5, 11]
    for(int i = 0; i<size_out; i++)
        std::cout << y[i] << " ";
    std::cout << std::endl;

    x.to("cuda");
    linear.to("cuda");
    Tensor y_cuda = linear.forward(x);
    y_cuda.to("cpu");
    std::cout << y_cuda[0] << " " << y_cuda[1] << std::endl;
    std::cout << std::endl;
}

void check_softmax() {
    std::cout << " check softmax " << std::endl << std::endl;
    size_t size = 4;
    Softmax softmax(size);

    Tensor x1(1, 4, "cpu", {1}, {4});
    x1[0] = 20.0f; x1[1] = 30.0f; x1[2] = 30.0f; x1[3] = 20.0f;
    x1.to("cuda");
    softmax.to("cuda");
    softmax.forward(x1);
    x1.to("cpu");
    std::cout << x1[0] << " " << x1[1] << " " << x1[2] << " " << x1[3] << std::endl;

    
    Tensor x(1, 4, "cpu", {1}, {4});
    x[0] = 20; x[1] = 30; x[2] = 30; x[3] = 20;
    softmax.to("cpu");
    softmax.forward(x);
    std::cout << x[0] << " " << x[1] << " " << x[2] << " " << x[3] << std::endl;
    std::cout << std::endl;
}

void check_embedding() {
    std::cout << " check embedding " << std::endl << std::endl;
    size_t vocal_size = 10;
    size_t hidden_size = 2;
    Embedding embedding(vocal_size, hidden_size);
    std::unordered_map<std::string, std::shared_ptr<float[]>> weights;
    weights["weight"] = std::shared_ptr<float[]>(new float[vocal_size*hidden_size]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                                                                11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f});
    embedding.load_weights(weights);

    Tensor x(2, 1, "cpu", {1}, {2});
    x[0] = 3; x[1] = 7;
    Tensor y = embedding.forward(x);
    for(int i = 0; i < y.elemNum(); i++) {
        for(int j = 0; j < y.elemLen(); j++)
            std::cout << y[i*y.elemLen() + j] << " ";
        std::cout << std::endl;
    }

    x.to("cuda");
    embedding.to("cuda");
    Tensor y_cuda = embedding.forward(x);
    y_cuda.to("cpu");
    for(int i = 0; i < y_cuda.elemNum(); i++) {
        for(int j = 0; j < y_cuda.elemLen(); j++)
            std::cout << y_cuda[i*y_cuda.elemLen() + j] << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void check_rsmnorm() {
    std::cout << " check rsm_norm " << std::endl << std::endl;
    size_t hidden_size = 4;
    std::unordered_map<std::string, std::shared_ptr<float[]>> weights;
    weights["weight"] = std::shared_ptr<float[]>(new float[hidden_size]{1.0f, 2.0f, 3.0f, 4.0f});

    RMSNorm rms_norm(hidden_size);
    rms_norm.load_weights(weights);

    Tensor x(1, hidden_size, "cpu", {1}, {2});
    x[0] = 1; x[1] = 2; x[2] = 3; x[3] = 4;
    Tensor y = rms_norm.forward(x);
    // x [1, 2]
    // w [[1,2],[3,4]] 列主导
    // y [5, 11]
    for(int i = 0; i < hidden_size; i++)
        std::cout << y[i] << " ";
    std::cout << std::endl;

    x[0] = 1; x[1] = 2; x[2] = 3; x[3] = 4;
    x.to("cuda");
    rms_norm.to("cuda");
    Tensor y_cuda = rms_norm.forward(x);
    y_cuda.to("cpu");
    for(int i = 0; i < hidden_size; i++)
        std::cout << y_cuda[i] << " ";
    std::cout << std::endl;
    std::cout << std::endl;
}

void check_rope() {
    std::cout << " check rotary positional embedding... " << std::endl << std::endl;
    size_t head_dim = 4;
    size_t hidden_size = head_dim * 2;
    size_t elem_num = 5;
    RoPE rope(head_dim);

    Tensor x(elem_num, hidden_size, "cpu", {}, {});
    std::vector<size_t> pos;
    for(int j = 0; j < elem_num; j++) {
        for(int i = 0; i < hidden_size; i++)
            x[j*hidden_size + i] = 1;
        pos.push_back(j);
    }
    x.addPos({pos});

    Tensor y = rope.forward(x);

    std::cout << "y: ";
    for(int j = 0; j < y.elemNum(); j++) {
        for(int i = 0; i < y.elemLen(); i++)
            std::cout << y[j * y.elemLen() + i] << " " ;
        std::cout << std::endl;
    }

    for(int j = 0; j < elem_num; j++) {
        for(int i = 0; i < hidden_size; i++)
            x[j*hidden_size + i] = 1;
    }
    x.to("cuda");
    rope.to("cuda");
    rope.forward(x);
    x.to("cpu");

    std::cout << "y_cuda: ";
    for(int j = 0; j < x.elemNum(); j++) {
        for(int i = 0; i < x.elemLen(); i++)
            std::cout << x[j * x.elemLen() + i] << " " ;
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void check_layerList() {
    std::cout << " check layerList " << std::endl << std::endl;
    size_t size_in = 2;
    size_t size_out = 2;
    std::unordered_map<std::string, std::shared_ptr<float[]>> weights;
    

    LayerList layer_list;
    Linear* linear1 = new Linear(size_in, size_out);
    weights["weight"] = std::shared_ptr<float[]>(new float[4]{1.0f, 2.0f, 3.0f, 4.0f});
    linear1->load_weights(weights);
    layer_list.add_layer(linear1, "layer1");
    Linear* linear2 = new Linear(size_in, size_out);
    weights["weight"] = std::shared_ptr<float[]>(new float[4]{1.0f, 2.0f, 3.0f, 4.0f});
    linear2->load_weights(weights);
    layer_list.add_layer(linear2, "layer2");

    Tensor x(1, 2, "cpu", {1}, {2});
    x[0] = 1; x[1] = 2;
    Tensor y = layer_list.forward(x);
    // x [1, 2]
    // w [[1,2],[3,4]] 列主导
    // y [5, 11]
    for(int i = 0; i<size_out; i++)
        std::cout << y[i] << " ";
    std::cout << std::endl;

    x.to("cuda");
    layer_list.to("cuda");
    Tensor y_cuda = layer_list.forward(x);
    y_cuda.to("cpu");
    std::cout << y_cuda[0] << " " << y_cuda[1] << std::endl;
    std::cout << std::endl;
}

void check_mlp() {
    std::cout << " check mlp " << std::endl << std::endl;
    size_t in_size = 2;
    size_t middle_size = 3;
    std::unordered_map<std::string, std::shared_ptr<float[]>> weights;
    weights["mlp.gate_proj.weight"] = std::shared_ptr<float[]>(new float[in_size*middle_size]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    weights["mlp.up_proj.weight"] = std::shared_ptr<float[]>(new float[in_size*middle_size]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    weights["mlp.down_proj.weight"] = std::shared_ptr<float[]>(new float[in_size*middle_size]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    
    Mlp mlp(in_size, middle_size);
    mlp.load_state(weights);

    Tensor x(1, in_size, "cpu", {1}, {in_size});
    x[0] = 1; x[1] = 2;
    Tensor y = mlp.forward(x);
    // x [1, 2]
    // w [[1,2],[3,4]] 列主导
    // y [5, 11]
    for(int i = 0; i<in_size; i++)
        std::cout << y[i] << " ";
    std::cout << std::endl;

    x.to("cuda");
    mlp.to("cuda");
    Tensor y_cuda = mlp.forward(x);
    y_cuda.to("cpu");
    for(int i = 0; i<in_size; i++)
        std::cout << y_cuda[i] << " ";
    std::cout << std::endl;
}

void check_attention() {
    std::cout << " check attention " << std::endl << std::endl;

    Config config("config.json");

    std::vector<std::vector<size_t>> input_ids = {{128000, 791, 1401, 311, 2324, 374}};
    std::vector<std::vector<size_t>> position = {{0, 1, 2, 3, 4, 5}};
    std::vector<size_t> uid = {112358};

    size_t vocal_size = 128256;
    size_t hidden_size = 2048;

    std::shared_ptr<float []> ptr(new float[6*hidden_size]);
    read_bin(ptr.get(), 6*hidden_size, "embed_token.bin");

    Tensor x(6, hidden_size, "cpu", uid, {6});
    x.addPos(position);
    x.setUid(uid);
    x.setValue(ptr);

    if(compare_results(ptr.get(), x, x.Size())) {
        check_pass("embedding check pass");
    } else {
        check_error("embedding check not pass");
    }

    DecoderLayer decoder(config, "model.layers.0");
    decoder.load_state("model.safetensors");
    x = decoder.forward(x);

    float* p = new float[6*hidden_size];
    read_bin(p, 6*hidden_size, "decoder_0.bin");
    
    if(compare_results(p, x, x.Size(), 5e-3)) {
        check_pass("decoder_0 check pass");
    } else {
        check_error("decoder_0 check not pass");
    }
}

int main() {

    // check_linear();
    // check_softmax();
    // check_embedding();
    // check_rope();
    // check_rsmnorm();
    // check_layerList();
    // check_mlp();
    check_attention();
    // Config config("config.json");

    // const size_t hidden_state = 2048;
    // const size_t vocab_size = 128256;
    // Llama llama = Llama(config);
    // std::cout << "loading model..." << std::endl;
    // llama.load_state("./model.safetensors", true);
    // std::cout << "loaded model" << std::endl;


    // std::vector<std::vector<size_t>> input_ids = {{128000, 791, 1401, 311, 2324, 374}};
    // Tensor x({input_ids});
    // x.setName("input_ids");
   
    // // 根据 tensor 产生 position
    // Tensor pos(x, 1);

    // for(int i = 0, index = 0; i < x.batchSize(); i++) {
    //     for(int j = 0; j < x.Seq()[i]; j++, index++) {
    //         pos[index] = j;
    //     }
    // }
    // Tensor y(x.batchSize(), 1, x.Device(), x.Uid());
    // llama.forward(y, x, pos);
    // std::cout << y[0] << std::endl;

    return 0;
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
            //break;
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
