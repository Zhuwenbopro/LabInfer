#ifndef ATTENTION_H
#define ATTENTION_H

#include "Layer.h"
#include "Linear.h"
#include "RoPE.h"
#include "RMSNorm.h"
#include "Config.h"
#include "Cache.h"

class Attention : public Layer {
public:
    Attention() = delete;
    Attention(const size_t attn_head, const size_t kv_head, const size_t hidden_size, const size_t _max_len = 250);

    void forward(InputWarp& inputWarp) override;

    void to(const std::string& _device) override;

    virtual ~Attention() = default;

private:
    size_t head_dim;
    size_t attn_head;
    size_t kv_head;
    size_t hidden_size;
    size_t q_dim;
    size_t kv_dim;

    Cache k_cache;
    Cache v_cache;
    // FIXME：max_len to be dynamic
    size_t max_len = 250;
};

Attention::Attention(
    const size_t _attn_head, 
    const size_t _kv_head, 
    const size_t _hidden_size,
    const size_t _max_len
) : Layer("cpu", "self_attn"), 
    head_dim(_hidden_size/_attn_head), 
    attn_head(_attn_head), 
    kv_head(_kv_head), 
    hidden_size(_hidden_size), 
    k_cache(Cache(_hidden_size, _max_len)),
    v_cache(Cache(_hidden_size, _max_len))
{

    q_dim = head_dim*_attn_head; 
    kv_dim = head_dim*_kv_head;
    layers.emplace("q_linear", new Linear(hidden_size, q_dim, "q_linear"));
    layers.emplace("k_linear", new Linear(hidden_size, kv_dim, "k_linear"));
    layers.emplace("v_linear", new Linear(hidden_size, kv_dim, "v_linear"));
    layers.emplace("o_linear", new Linear(q_dim, hidden_size, "o_linear"));
    layers.emplace("rope", new RoPE(head_dim));
}


void _read_bin(const std::string& filename, float* ptr, size_t size) {
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



void _check_pass(const char*  message){
    std::cout << "\033[32m" << message << "\033[0m" << std::endl;
}

void _check_error(const char*  message){
    std::cout << "\033[31m" << message << "\033[0m" << std::endl;
}

bool _compare_results(const float *a, const float *b, int size, float tolerance) {
    for (int i = 0; i < size; ++i) {
        if (std::fabs(a[i] - b[i]) > tolerance) {
            std::cout << "Difference at index " << i << ": " << a[i] << " vs " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

void _check(const float *a, const float *b, int size, const std::string& item, float tolerance=1e-2f) {
    if (_compare_results(a, b, size, 5e-5)) {
        _check_pass(("[" + item + "] CUDA and CPU results match.").c_str());
    } else {
        _check_error(("[" + item + "] CUDA and CPU results do not match!").c_str());
    }

    for(int i = 0; i < 5; i++) {
        if(i >= size) break;
        std::cout << a[i] << " vs " << b[i] << std::endl;
    }
}


void Attention::forward(InputWarp& inputWarp) {
    size_t uid = inputWarp.uid;
    Tensor<float> x   = inputWarp.inter_value;

    Tensor<float> q = layers.at("q_linear")->forward(x);
    Tensor<float> k = layers.at("k_linear")->forward(x);
    Tensor<float> v = layers.at("v_linear")->forward(x);

    q = layers.at("rope")->forward(q, inputWarp.pos);
    k = layers.at("rope")->forward(k, inputWarp.pos);

    int rep = attn_head/kv_head;
    if(rep != 1) {
        Tensor<float> k_exp(x.ElemNum(), q.ElemLen(), device, "k_exp");
        Tensor<float> v_exp(x.ElemNum(), q.ElemLen(), device, "v_exp");
        F->repeat_kv(k_exp, k, head_dim, attn_head/kv_head, k.Size());
        F->repeat_kv(v_exp, v, head_dim, attn_head/kv_head, k.Size());
        k = k_exp;
        v = v_exp;
    }
    k_cache.add(uid, k, inputWarp.start_pos);
    v_cache.add(uid, v, inputWarp.start_pos);

    Tensor<float> o(q.ElemNum(), q.ElemLen(), q.Device(), "attn_output");

    F->masked_attention(o, q, k_cache.get(uid), v_cache.get(uid), nullptr, inputWarp.pos, head_dim, attn_head, q.ElemNum(), k_cache.Len(uid));

    inputWarp.inter_value = layers.at("o_linear")->forward(o);

    inputWarp.inter_value.to(x.Device());
}

void Attention::to(const std::string& _device) {
    if(device == _device) return;

    for (auto& [_name, param] : params) {
        param.to(_device);
    }

    F = DeviceManager::getInstance().getDevice(_device)->F;

    for (auto& [_name, layer] : layers) {
        layer->to(_device);
    }

    device = _device;

    k_cache.to(_device);
    v_cache.to(_device);
}

#endif // ATTENTION_H