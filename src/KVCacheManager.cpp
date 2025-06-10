#include "KVCacheManager.h"
#include "CPUMemoryManager.h"
#include "CUDAMemoryManager.h"
#include <iostream>
#include <algorithm>

KVCacheManager::KVCacheManager(MemoryType memory_type, size_t max_cache_size, 
                               bool enable_paged_attention, size_t page_size)
    : page_size_(page_size)
    , paged_attention_enabled_(enable_paged_attention)
{
    // 根据内存类型创建对应的内存管理器
    switch (memory_type) {
        case MemoryType::CPU:
            memory_manager_ = std::make_unique<CPUMemoryManager>();
            break;
        case MemoryType::CUDA:
            memory_manager_ = std::make_unique<CUDAMemoryManager>();
            break;
        default:
            memory_manager_ = std::make_unique<CPUMemoryManager>();
            break;
    }
    
    // 计算最大页数
    max_pages_ = max_cache_size / page_size_;
}

KVCacheManager::~KVCacheManager() {
    // 清理所有缓存
    clear();
}

bool KVCacheManager::allocate(const std::string& request_id, int layer_idx, 
                             size_t key_size, size_t value_size, int head_idx) {
    // 检查是否已存在该缓存
    auto req_it = kv_cache_map_.find(request_id);
    if (req_it != kv_cache_map_.end()) {
        auto layer_it = req_it->second.find(layer_idx);
        if (layer_it != req_it->second.end()) {
            // 已存在该层的缓存，先释放
            free(request_id, layer_idx);
        }
    }
    
    try {
        // 分配内存
        void* key_cache = memory_manager_->allocate(key_size);
        void* value_cache = memory_manager_->allocate(value_size);
        
        if (!key_cache || !value_cache) {
            // 分配失败，释放已分配的内存
            if (key_cache) {
                memory_manager_->deallocate(key_cache);
            }
            if (value_cache) {
                memory_manager_->deallocate(value_cache);
            }
            return false;
        }
        
        // 创建缓存条目
        KVCacheEntry entry;
        entry.key_cache = key_cache;
        entry.value_cache = value_cache;
        entry.size = key_size + value_size;
        entry.layer_idx = layer_idx;
        entry.head_idx = head_idx;
        entry.is_paged = paged_attention_enabled_;
        
        // 添加到映射表
        kv_cache_map_[request_id][layer_idx] = entry;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error allocating KV cache: " << e.what() << std::endl;
        return false;
    }
}

const KVCacheManager::KVCacheEntry* KVCacheManager::get(const std::string& request_id, int layer_idx) const {
    auto req_it = kv_cache_map_.find(request_id);
    if (req_it == kv_cache_map_.end()) {
        return nullptr;
    }
    
    auto layer_it = req_it->second.find(layer_idx);
    if (layer_it == req_it->second.end()) {
        return nullptr;
    }
    
    return &(layer_it->second);
}

bool KVCacheManager::free(const std::string& request_id) {
    auto req_it = kv_cache_map_.find(request_id);
    if (req_it == kv_cache_map_.end()) {
        return false;
    }
    
    // 释放该请求的所有缓存
    for (auto& layer_entry : req_it->second) {
        memory_manager_->deallocate(layer_entry.second.key_cache);
        memory_manager_->deallocate(layer_entry.second.value_cache);
    }
    
    // 从映射表中移除
    kv_cache_map_.erase(req_it);
    
    return true;
}

bool KVCacheManager::free(const std::string& request_id, int layer_idx) {
    auto req_it = kv_cache_map_.find(request_id);
    if (req_it == kv_cache_map_.end()) {
        return false;
    }
    
    auto layer_it = req_it->second.find(layer_idx);
    if (layer_it == req_it->second.end()) {
        return false;
    }
    
    // 释放该层的缓存
    memory_manager_->deallocate(layer_it->second.key_cache);
    memory_manager_->deallocate(layer_it->second.value_cache);
    
    // 从映射表中移除
    req_it->second.erase(layer_it);
    
    // 如果该请求没有其他缓存了，也移除请求条目
    if (req_it->second.empty()) {
        kv_cache_map_.erase(req_it);
    }
    
    return true;
}

bool KVCacheManager::extend(const std::string& request_id, int layer_idx, 
                           size_t new_key_size, size_t new_value_size) {
    auto req_it = kv_cache_map_.find(request_id);
    if (req_it == kv_cache_map_.end()) {
        return false;
    }
    
    auto layer_it = req_it->second.find(layer_idx);
    if (layer_it == req_it->second.end()) {
        return false;
    }
    
    // 获取当前缓存条目
    KVCacheEntry& entry = layer_it->second;
    
    // 分配新的内存
    void* new_key_cache = memory_manager_->allocate(new_key_size);
    void* new_value_cache = memory_manager_->allocate(new_value_size);
    
    if (!new_key_cache || !new_value_cache) {
        // 分配失败，释放已分配的内存
        if (new_key_cache) {
            memory_manager_->deallocate(new_key_cache);
        }
        if (new_value_cache) {
            memory_manager_->deallocate(new_value_cache);
        }
        return false;
    }
    
    // 释放旧的内存
    memory_manager_->deallocate(entry.key_cache);
    memory_manager_->deallocate(entry.value_cache);
    
    // 更新缓存条目
    entry.key_cache = new_key_cache;
    entry.value_cache = new_value_cache;
    entry.size = new_key_size + new_value_size;
    
    return true;
}

void KVCacheManager::clear() {
    // 释放所有缓存
    for (auto& req_entry : kv_cache_map_) {
        for (auto& layer_entry : req_entry.second) {
            memory_manager_->deallocate(layer_entry.second.key_cache);
            memory_manager_->deallocate(layer_entry.second.value_cache);
        }
    }
    
    // 清空映射表
    kv_cache_map_.clear();
}

size_t KVCacheManager::get_used_size() const {
    size_t total_size = 0;
    
    for (const auto& req_entry : kv_cache_map_) {
        for (const auto& layer_entry : req_entry.second) {
            total_size += layer_entry.second.size;
        }
    }
    
    return total_size;
}

size_t KVCacheManager::get_max_size() const {
    return max_pages_ * page_size_;
}

void KVCacheManager::set_max_size(size_t size) {
    max_pages_ = size / page_size_;
}

void KVCacheManager::enable_paged_attention(bool enable, size_t page_size) {
    paged_attention_enabled_ = enable;
    
    if (enable && page_size > 0) {
        page_size_ = page_size;
        // 重新计算最大页数
        max_pages_ = get_max_size() / page_size_;
    }
}

size_t KVCacheManager::get_page_size() const {
    return page_size_;
}

KVCacheManager::MemoryType KVCacheManager::get_memory_type() const {
    // 根据当前内存管理器类型返回对应的枚举值
    if (dynamic_cast<CPUMemoryManager*>(memory_manager_.get())) {
        return MemoryType::CPU;
    } else if (dynamic_cast<CUDAMemoryManager*>(memory_manager_.get())) {
        return MemoryType::CUDA;
    }
    
    // 默认返回CPU类型
    return MemoryType::CPU;
}

// bool KVCacheManager::switch_memory_type(MemoryType memory_type) {
//     // 如果类型相同，无需切换
//     if (get_memory_type() == memory_type) {
//         return true;
//     }
    
//     // 创建新的内存管理器
//     std::unique_ptr<MemoryManager> new_manager;
    
//     switch (memory_type) {
//         case MemoryType::CPU:
//             new_manager = std::make_unique<CPUMemoryManager>();
//             break;
//         case MemoryType::CUDA:
//             new_manager = std::make_unique<CUDAMemoryManager>();
//             break;
//         default:
//             return false;
//     }
    
//     // 保存当前缓存状态
//     std::unordered_map<std::string, std::unordered_map<int, KVCacheEntry>> old_cache_map = kv_cache_map_;
    
//     // 清空当前缓存
//     clear();
    
//     // 切换内存管理器
//     memory_manager_ = std::move(new_manager);
    
//     // 重新分配所有缓存（这里简化处理，实际可能需要数据迁移）
//     bool success = true;
//     for (const auto& req_entry : old_cache_map) {
//         for (const auto& layer_entry : req_entry.second) {
//             const KVCacheEntry& entry = layer_entry.second;
//             size_t key_size = entry.size / 2;  // 简化处理，假设key和value大小相同
//             size_t value_size = entry.size - key_size;
            
//             if (!allocate(req_entry.first, entry.layer_idx, key_size, value_size, entry.head_idx)) {
//                 success = false;
//             }
//         }
//     }
    
//     return success;
// }

bool KVCacheManager::is_paged_attention_enabled() const {
    return paged_attention_enabled_;
}

size_t KVCacheManager::get_cache_count(const std::string& request_id) const {
    auto req_it = kv_cache_map_.find(request_id);
    if (req_it == kv_cache_map_.end()) {
        return 0;
    }
    
    return req_it->second.size();
}

std::vector<std::string> KVCacheManager::get_all_request_ids() const {
    std::vector<std::string> request_ids;
    request_ids.reserve(kv_cache_map_.size());
    
    for (const auto& req_entry : kv_cache_map_) {
        request_ids.push_back(req_entry.first);
    }
    
    return request_ids;
}