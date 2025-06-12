#pragma once
#include <map>
#include <tuple>
#include <string>
#include <stdexcept>
#include "common.h"
#include "function.h"

template <typename FuncPtrT>
class FuncRegistry {
public:
    void Register(int device, int dtype, FuncPtrT ptr) {
        registry_map_[std::make_tuple(device, dtype)] = ptr;
    }

    FuncPtrT Get(int device, int dtype) const {
        auto it = registry_map_.find(std::make_tuple(device, dtype));
        if (it != registry_map_.end()) {
            return it->second;
        }
        // 更友好的错误提示
        throw std::runtime_error("Function implementation not found for device/dtype combination.");
        return nullptr;
    }

private:
    std::map<std::tuple<int, int>, FuncPtrT> registry_map_;
};

// 全局的注册表管理器单例
// 它为每种操作类型持有一个专门的注册表实例
class OpRegistry {
public:
    static OpRegistry& Instance() {
        static OpRegistry instance;
        return instance;
    }
    
    // 禁止拷贝和赋值
    OpRegistry(const OpRegistry&) = delete;
    OpRegistry& operator=(const OpRegistry&) = delete;

    // 提供对特定类型函数指针注册表的访问
    FuncRegistry<LinearFuncPtr>& Linear() { return linear_registry_; }
    // FuncRegistry<AttentionFuncPtr>& Attention() { return attention_registry_; }
    // FuncRegistry<RMSNormFuncPtr>& RMSNorm() { return rmsnorm_registry_; }
    // ...

private:
    OpRegistry() = default;

    FuncRegistry<LinearFuncPtr> linear_registry_;
    // FuncRegistry<AttentionFuncPtr> attention_registry_;
    // FuncRegistry<RMSNormFuncPtr> rmsnorm_registry_;
    // ...
};

// 通用的注册宏
#define REGISTER_OP_FUNCTION(OP_NAME, DEVICE, DTYPE, FUNC_PTR) \
    namespace { \
        struct Registrar_##OP_NAME##_##DEVICE##_##DTYPE { \
            Registrar_##OP_NAME##_##DEVICE##_##DTYPE() { \
                OpRegistry::Instance().OP_NAME().Register(DEVICE, DTYPE, FUNC_PTR); \
            } \
        }; \
        static Registrar_##OP_NAME##_##DEVICE##_##DTYPE registrar_instance; \
    }
