#include "Cache.h"

Cache::Cache(const size_t _len, const size_t _max_len, const std::string& dev) : len(_len), max_len(_max_len), device(dev) { }

Cache::Cache() : len(0), max_len(0), device("cpu") { }

Cache::~Cache() { }

float* Cache::get(const size_t uid) {
    auto it = caches.find(uid);
    if (it!= caches.end()) {
        return (it->second);
    } else {
        throw std::logic_error("Didn't get uid in caches " + uid); 
        return nullptr;
    }
}

void Cache::add(Tensor& t) {
    if(t.elemLen() != len)
        throw std::logic_error("Cache elemlen do not match tensor elemlen.");

    int index_offset = 0;
    // 对
    for(int i = 0; i < t.Uid().size(); i++) {
        auto uid = t.Uid()[i];
        auto it = caches.find(uid);
        if (it== caches.end()) { // 没有这个 uid
            caches.emplace(uid, Parameter(" ", max_len, len, device, true));
        }

        auto num = t.SeqLen()[i];
        size_t offset = t.Position()[i][0] * len;

        // 写一个parameter copy函数，把值考进来    
        if(device != t.Device()) 
            throw std::logic_error("copy value not on the same device.");

        caches.at(uid).copy(t, num * len, index_offset * len, offset);
        index_offset += num;
    }
}

void Cache::to(const std::string& new_dev) {
    for (auto& [key, parameter] : caches) {
        parameter.to(new_dev);
    }
    device = new_dev;
}

