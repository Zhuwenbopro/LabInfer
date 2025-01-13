#ifndef CACHE_H
#define CACHE_H

#include <unordered_map>
#include "Parameter.h"
#include "Tensor.h"

class Cache {
private:
    std::string device;
    size_t max_len;
    size_t len;
    // FIXME : 这里未来要把 float 给解耦
    std::unordered_map<size_t, Parameter<float>> caches;
    std::unordered_map<size_t, size_t> cache_lens;

public:
    Cache(const size_t _len, const size_t _max_len) : len(_len), max_len(_max_len), device("cpu") { }

    ~Cache() { }

    // FIXME : 这里未来要把 float 给解耦
    Parameter<float> get(const size_t uid) {
        auto it = caches.find(uid);
        // 不存在
        if (it == caches.end()) {
            throw std::logic_error("no " + std::to_string(uid) + " in cache\n"); 
        }
        return caches[uid];
    }

    size_t Len(const size_t uid) {
        auto it = cache_lens.find(uid);
        // 不存在
        if (it == cache_lens.end()) {
            throw std::logic_error("no " + std::to_string(uid) + " in cache\n"); 
        }
        return cache_lens[uid];
    }

    void add(const size_t uid, Tensor<float>& tensor, const int start_pos) {
        
        if(tensor.ElemLen() != len) {
            throw std::logic_error("tensor " + std::to_string(uid) + " EleLen does not match cache len!\n"); 
        }
        
        auto it = caches.find(uid);
        // 不存在
        if (it == caches.end()) {
            caches[uid] = Parameter<float>(max_len, len, device, std::to_string(uid), true);
            cache_lens[uid] = 0;
        }
        
        caches[uid].copy(start_pos*len, tensor, 0, tensor.Size());
        cache_lens[uid] = start_pos + tensor.ElemNum();
    }

    void clear(const size_t uid) {
        caches.erase(uid);
        cache_lens.erase(uid);
    }

    void to(const std::string& _device) {
        for (auto& pair : caches) {
            pair.second.to(_device);
        }
        device = _device;
    }

    void print() {
        std::cout << "len = " << len << std::endl;
        std::cout << "max_len = " << max_len << std::endl;
        std::cout << "device = " << device << std::endl;
    }
};

#endif