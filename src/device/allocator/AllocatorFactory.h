// AllocatorFactory.h
#ifndef ALLOCATOR_FACTORY_H
#define ALLOCATOR_FACTORY_H

#include "Allocator.h"
#include <stdexcept>
#include <string>
#include <unordered_map>

class AllocatorFactory {
public:
    // 获取单例工厂实例
    static AllocatorFactory instance() {
        static AllocatorFactory factory;
        return factory;
    }

    // 注册类
    bool registerFactory(const std::string& name, Allocator* allocator) {
        auto it = allocators.find(name);
        if (it == allocators.end()) {
            allocators[name] = allocator;
            return true;
        }
        return std::runtime_error("Unknown Allocator type: " + name);
    }

    Allocator* getAllocator() {
        auto it = allocators.find(name);
        if (it == allocators.end()) {
            return NULL;
        }
        return it->second;
    }
private:
    AllocatorFactory() = default;
    std::unordered_map<std::string, Allocator*> allocators;
}

#endif // ALLOCATOR_FACTORY_H