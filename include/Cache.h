#ifndef CACHE_H
#define CACHE_H

#include <vector>
#include <unordered_map>
#include <memory>
#include "Tensor.h"
#include "Parameter.h"
#include "Manager.h"
#include "Config.h"

class Cache
{
private:
    std::string device;
    size_t max_len;
    size_t len;
    std::unordered_map<size_t, Parameter> caches;

public:
    Cache(const size_t _len, const size_t max_len, const std::string& dev);
    Cache();
    ~Cache();
    float* get(const size_t uid);
    void add(Tensor& ptr);
    void to(const std::string& new_dev);
};

#endif