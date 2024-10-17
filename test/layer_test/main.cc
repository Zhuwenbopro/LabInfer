#include "layers.h"
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
void rand_init(float* ptr, int size);
void const_init(float* ptr, int size);

void check_embedding();
void check_linear();
void check_softmax();
void cheack_transform();

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

int main() {

    // RoPE rope(64);
    // check_linear();
    // check_softmax();
    // check_embedding();
    // check_attention();
    cheack_transform();

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

void rand_init(float* ptr, int size){
    // 设置随机数种子
    std::srand(static_cast<unsigned int>(std::time(0)));

    for (int i = 0; i < size; ++i) {
        ptr[i] = static_cast<float>(rand()) / RAND_MAX;
    }    
}

void const_init(float* ptr, int size) {
    for (int i = 0; i < size; ++i) {
        ptr[i] = i;
    }    
}

void check_linear() {
    size_t size_in = 4096;
    size_t size_out = 2048;

    Linear linear = Linear(size_in, size_out);

    Parameter W("Linear.weight", {size_in, size_out}, "cpu", true);
    // 初始化输入数据和权重
    rand_init(W.rawPtr(), size_in * size_out);

    Tensor x("input", {size_in}, "cpu", true);
    Tensor y_cpu("output cpu", {size_out}, "cpu", true);
    rand_init(x.rawPtr(), size_in);

    std::unordered_map<std::string, std::shared_ptr<float []>> states;
    states["Linear.weight"] = W.sharedPtr();
    linear.load_state(states);

    auto start = std::chrono::high_resolution_clock::now();
    linear.forward(y_cpu, x);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Execution time: " << duration << " microseconds" << std::endl;
    Tensor y_cuda = y_cpu.copy();
    y_cuda.setName("op cuda");

    y_cuda.to("cuda");
    x.to("cuda");
    linear.to("cuda");
    start = std::chrono::high_resolution_clock::now();
    linear.forward(y_cuda, x);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Execution time: " << duration << " microseconds" << std::endl;

    y_cuda.to("cpu");
    if (compare_results(y_cuda, y_cpu, size_out, 1e-2)) {
        check_pass("[linear] CUDA and CPU results match.");
    } else {
        check_error("[linear] CUDA and CPU results do not match!");
    }

}

void check_softmax() {
    size_t size_in = 4096;
    Softmax softmax = Softmax(size_in, "Softmax");

    // 初始化输入数据和权重
    Tensor x_cpu("input", {size_in}, "cpu", true);
    rand_init(x_cpu.rawPtr(), size_in);
    Tensor x_cuda = x_cpu.copy();

    auto start = std::chrono::high_resolution_clock::now();
    softmax.forward(x_cpu);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Execution time: " << duration << " microseconds" << std::endl;

    x_cuda.to("cuda");
    softmax.to("cuda");
    start = std::chrono::high_resolution_clock::now();
    softmax.forward(x_cuda);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Execution time: " << duration << " microseconds" << std::endl;

    x_cuda.to("cpu");

    if (compare_results(x_cuda, x_cpu, size_in)) {
        check_pass("[softmax] CUDA and CPU results match.");
    } else {
        check_error("[softmax] CUDA and CPU results do not match!");
    }
}

void check_embedding() {
    std::cout << std::endl << "begin test embedding" << std::endl;
    const size_t dim0 = 128256;
    const size_t dim1 = 2048;
    const size_t total_elements = dim0 * dim1;
    int output_size = 6 * dim1;
    
    // 分配主机内存
    Parameter weight("weight", {dim1, dim0}, "cpu", true);
    read_bin(weight, total_elements, "model_embed_tokens_weight.bin");


    Tensor result("result", {6 * dim1}, "cpu", true);
    read_bin(result, output_size, "embedding_tensor.bin");   

    Tensor x("embedding input", {1, 6}, "cpu", true);
    x[0] = 128000;
    x[1] = 791;
    x[2] = 1401;
    x[3] = 311;
    x[4] = 2324;
    x[5] = 374;

    Embedding embedding = Embedding(dim0, dim1);
    std::unordered_map<std::string, std::shared_ptr<float []>> states;
    states["embed_tokens.weight"] = weight.sharedPtr();
    embedding.load_state(states);

    Tensor y_cpu("embedding output", {6, dim1}, "cpu", true);
    Tensor y_cuda = y_cpu.copy();

    auto start = std::chrono::high_resolution_clock::now();
    embedding.forward(y_cpu, x);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Execution time: " << duration << " microseconds" << std::endl;

    y_cuda.to("cuda");
    y_cuda.setName("y cuda");
    x.to("cuda");
    embedding.to("cuda");

    start = std::chrono::high_resolution_clock::now();
    embedding.forward(y_cuda, x);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Execution time: " << duration << " microseconds" << std::endl;

    y_cuda.to("cpu");


    if (compare_results(result, y_cpu, output_size)) {
        check_pass("[embedding] CPU results correct.");
    } else {
        check_error("[embedding] CPU results error!");
    }

    if (compare_results(y_cuda, result, output_size)) {
        check_pass("[embedding] CUDA results correct.");
    } else {
        check_error("[embedding] CUDA results error!");
    }
}

void test_layer(Layer& layer, Tensor& result, Tensor& x, Tensor& real_result, float e = 1e-2) {
    x.to(layer.Device());
    result.to(layer.Device());

    auto start = std::chrono::high_resolution_clock::now();
    layer.forward(result, x);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Execution time: " << duration << " microseconds" << std::endl;

    if(layer.Device() != "cpu")
        result.to("cpu");
    
    if (compare_results(result, real_result, result.Size(), e)) {
        check_pass("[" + layer.Name() +"] " + layer.Device() + " results correct.");
    } else {
        check_error("[" + layer.Name() +"] " + layer.Device() + " results error!");
    }
}

void test_layer(Layer& layer, Tensor& result, Tensor& real_result, float e = 1e-2) {
    result.to(layer.Device());

    auto start = std::chrono::high_resolution_clock::now();
    layer.forward(result);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Execution time: " << duration << " microseconds" << std::endl;

    if(layer.Device() != "cpu")
        result.to("cpu");
    
    if (compare_results(result, real_result, result.Size(), e)) {
        check_pass("[" + layer.Name() +"] " + layer.Device() + " results correct.");
    } else {
        check_error("[" + layer.Name() +"] " + layer.Device() + " results error!");
    }
}

void load_layer(Layer& layer, const std::string filename, const std::vector<size_t>& weight_shape, const std::string map_name) {
    if(layer.Device() != "cpu"){
        
    }
    Parameter weight("weight", weight_shape, "cpu", true);
    read_bin(weight, weight.Size(), filename);
    states[map_name] = weight.sharedPtr();
    layer.load_state(states);
}

void cheack_transform() {
    std::cout << std::endl << "begin test attention transformer" << std::endl;

    size_t batch = 1;
    size_t seq = 6;
    size_t hidden_state = 2048;
    size_t kv_dim = 512;

    Tensor x_cpu("embedding", {batch, hidden_state}, "cpu", true, {seq});
    read_bin(x_cpu, x_cpu.Size(), "embedding_tensor.bin");

// =============================================== RMS Norm =================================================

    RMSNorm rms_norm = RMSNorm(hidden_state);
    load_layer(rms_norm, "model_layers_0_input_layernorm_weight.bin", {hidden_state}, "RMSNorm.weight");
    Tensor norm_result("result", {batch, hidden_state}, "cpu", true, {seq});
    read_bin(norm_result, seq * hidden_state, "input_layernorm_output.bin");
    test_layer(rms_norm, x_cpu, norm_result);

// ==========================================================================================================
// =========================================== q、k、v linear ===============================================

    Linear q_linear = Linear(hidden_state, hidden_state, "linear_q");
    load_layer(q_linear, "model_layers_0_self_attn_q_proj_weight.bin", {hidden_state, hidden_state}, "linear_q.weight");
    Tensor q("q", {batch, hidden_state}, "cpu", true, {seq});
    Tensor q_check("q", {batch, hidden_state}, "cpu", true, {seq});
    read_bin(q_check, q.Size(), "query_states.bin");
    test_layer(q_linear, q, x_cpu, q_check, 6e-2);
    
    Linear k_linear = Linear(hidden_state, kv_dim, "linear_k");
    load_layer(k_linear, "model_layers_0_self_attn_k_proj_weight.bin", {hidden_state, kv_dim}, "linear_k.weight");
    Tensor k("k_result_cpu", {batch, kv_dim}, "cpu", true, {seq});
    Tensor k_check("k_result_cpu", {batch, kv_dim}, "cpu", true, {seq});
    read_bin(k_check, k.Size(), "key_states.bin");
    test_layer(k_linear, k, x_cpu, k_check, 6e-2);
    
    Linear v_linear = Linear(hidden_state, kv_dim, "linear_v");
    load_layer(v_linear, "model_layers_0_self_attn_v_proj_weight.bin", {hidden_state, kv_dim}, "linear_v.weight");
    Tensor v("v_result_cpu", {batch, kv_dim}, "cpu", true, {seq});
    Tensor v_check("v_result_cpu", {batch, kv_dim}, "cpu", true, {seq});
    read_bin(v_check, v.Size(), "value_states.bin");
    test_layer(v_linear, v, x_cpu, v_check);

// ==========================================================================================================
// =================================== rotary positional embedding ==========================================

    RoPE rope(64);
    Tensor pos("position", {1, 6}, "cpu", true, {1});
    pos[0] = 0; pos[2] = 2; pos[4] = 4;
    pos[1] = 1; pos[3] = 3; pos[5] = 5;

    Tensor query_pos_check("q_result_cpu", q.Shape(), "cpu", true, q.Seq());
    read_bin(query_pos_check, q.Size(), "pos_query.bin");
    test_layer(rope, q, pos, query_pos_check, 6e-2);

    Tensor key_pos_check("k_result_cpu", k.Shape(), "cpu", true, k.Seq());
    read_bin(key_pos_check, k.Size(), "pos_key.bin");
    test_layer(rope, k, pos, key_pos_check, 6e-2);

// ==========================================================================================================
// ========================================= attention ======================================================

    Tensor o("o_input", q.Shape(), "cpu", true, q.Seq());
    Tensor o_check = o.copy();
    read_bin(o_check, o_check.Size(), "atten_output.bin");

    Manager& manager = Manager::getInstance();
    Function& F = manager.getFunction("cpu");

    for(int p = 0; p < seq; p++) {
        float* output = o + p * hidden_state;
        float* query = q + p * hidden_state;
        F.maksed_attention(output, query, k, v, 64, 32, 8, p);
    }

    if (compare_results(o, o_check, o.Size(), 6e-2)) {
        check_pass("[ T_T ] CPU attention results correct.");
    } else {
        check_error("[ T_T ] CPU attention results error!");
    }

// ==========================================================================================================
// ============================================ o linear ====================================================
    Tensor o_r = o.copy();
    Linear o_linear = Linear(hidden_state, hidden_state, "linear_o");
    load_layer(o_linear, "model_layers_0_self_attn_o_proj_weight.bin", {hidden_state, hidden_state}, "linear_o.weight");
    read_bin(o_check, o.Size(), "self_attn.bin");
    test_layer(o_linear, o_r, o, o_check, 6e-2);



    Attention attn("attention");
    Parameter qw("qw", {hidden_state,hidden_state}, "cpu", true);
    Parameter kw("kw", {hidden_state,512}, "cpu", true);
    Parameter vw("vw", {hidden_state,512}, "cpu", true);
    Parameter ow("ow", {hidden_state,hidden_state}, "cpu", true);
    Parameter rms("rms", {hidden_state}, "cpu", true);
    read_bin(qw, qw.Size(), "model_layers_0_self_attn_q_proj_weight.bin");
    read_bin(kw, kw.Size(), "model_layers_0_self_attn_k_proj_weight.bin");
    read_bin(vw, vw.Size(), "model_layers_0_self_attn_v_proj_weight.bin");
    read_bin(ow, ow.Size(), "model_layers_0_self_attn_o_proj_weight.bin");
    read_bin(rms, rms.Size(), "model_layers_0_input_layernorm_weight.bin");


    std::unordered_map<std::string, std::shared_ptr<float []>> states;
    states["attention.q_linear.weight"] = qw.sharedPtr();
    states["attention.k_linear.weight"] = kw.sharedPtr();
    states["attention.v_linear.weight"] = vw.sharedPtr();
    states["attention.o_linear.weight"] = ow.sharedPtr();
    states["attention.RMSNorm.weight"] = rms.sharedPtr();
    attn.load_state(states);

    Tensor x("embedding", {batch, hidden_state}, "cpu", true, {seq});
    Tensor x_cuda = x.copy();
    read_bin(x, x.Size(), "embedding_tensor.bin");
    Tensor y = x.copy();
    Tensor y_cuda = y.copy();

    attn.forward(y, x, pos);

    if (compare_results(y, o_check, o.Size(), 6e-2)) {
        check_pass("[ T_T ] CPU attention results correct.");
    } else {
        check_error("[ T_T ] CPU attention results error!");
    }

    x_cuda.to("cuda");
    attn.to("cuda");

    y_cuda.to("cuda");
    attn.forward(y_cuda, x_cuda, pos);
    y_cuda.to("cpu");
    if (compare_results(y, o_check, o.Size(), 6e-2)) {
        check_pass("[ T_T ] CUDA attention results correct.");
    } else {
        check_error("[ T_T ] CUDA attention results error!");
    }

    return;
}