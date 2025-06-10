#pragma once

#include "MemoryManager.h"
#include <unordered_map>
#include <string>
#include <memory>
#include <vector>

/**
 * @brief KV缓存管理器，负责管理模型推理过程中的KV缓存
 * 
 * 该类可以根据传入的符号选择不同的内存管理器类型（CPU或CUDA），
 * 并维护KV缓存的地址映射表。未来将支持Paged Attention的页式缓存管理。
 */
class KVCacheManager
{
public:
    /**
     * @brief 内存类型枚举
     */
    enum class MemoryType {
        CPU,
        CUDA
    };
    /**
     * @brief KV缓存条目结构
     */
    struct KVCacheEntry {
        void* key_cache;       // 键缓存的地址
        void* value_cache;     // 值缓存的地址
        size_t size;           // 缓存大小（字节）
        int layer_idx;         // 层索引
        int head_idx;          // 注意力头索引
        bool is_paged;         // 是否为页式缓存
    };
private:
    // 内存管理器
    std::unique_ptr<MemoryManager> memory_manager_;
    
    // KV缓存映射表：请求ID -> 层索引 -> KV缓存条目
    std::unordered_map<std::string, std::unordered_map<int, KVCacheEntry>> kv_cache_map_;
    
    // 页式缓存相关参数
    size_t page_size_;         // 页大小（字节）
    size_t max_pages_;         // 最大页数
    bool paged_attention_enabled_=false; // 是否启用页式缓存

public:
    /**
     * @brief 构造函数
     * @param memory_type 内存类型，决定使用CPU还是CUDA内存管理器
     * @param max_cache_size 最大缓存大小（字节）
     * @param enable_paged_attention 是否启用页式缓存管理
     * @param page_size 页大小（字节），仅在启用页式缓存时有效
     */
    KVCacheManager(MemoryType memory_type = MemoryType::CPU, // 默认分配CPU内存
        size_t max_cache_size = 1024 * 1024 * 1024,  // 默认1GB
        bool enable_paged_attention = false,
        size_t page_size = 4 * 1024);  // 默认4KB
    /**
     * @brief 析构函数
     */
    ~KVCacheManager();

    /**
     * @brief 分配KV缓存
     * @param request_id 请求ID
     * @param layer_idx 层索引
     * @param key_size 键缓存大小
     * @param value_size 值缓存大小
     * @param head_idx 注意力头索引，默认为-1表示所有头
     * @return 是否分配成功
     */
    bool allocate(const std::string& request_id, int layer_idx, 
        size_t key_size, size_t value_size, int head_idx = -1);

    /**
     * @brief 获取KV缓存条目
     * @param request_id 请求ID
     * @param layer_idx 层索引
     * @return KV缓存条目指针，如果不存在则返回nullptr
     */
    const KVCacheEntry* get(const std::string& request_id, int layer_idx) const;

    /**
     * @brief 释放指定请求的所有KV缓存
     * @param request_id 请求ID
     * @return 是否释放成功
     */
    bool free(const std::string& request_id);
    
    /**
     * @brief 释放指定请求的特定层的KV缓存
     * @param request_id 请求ID
     * @param layer_idx 层索引
     * @return 是否释放成功
     */
    bool free(const std::string& request_id, int layer_idx);
    
    /**
     * @brief 扩展KV缓存（用于序列长度增加时）
     * @param request_id 请求ID
     * @param layer_idx 层索引
     * @param new_key_size 新的键缓存大小
     * @param new_value_size 新的值缓存大小
     * @return 是否扩展成功
     */
    bool extend(const std::string& request_id, int layer_idx, 
               size_t new_key_size, size_t new_value_size);
    
    /**
     * @brief 清空所有KV缓存
     */
    void clear();
    
    /**
     * @brief 获取当前已使用的缓存总大小
     * @return 已使用的缓存大小（字节）
     */
    size_t get_used_size() const;
    
    /**
     * @brief 获取最大缓存大小
     * @return 最大缓存大小（字节）
     */
    size_t get_max_size() const;
    
    /**
     * @brief 设置最大缓存大小
     * @param size 最大缓存大小（字节）
     */
    void set_max_size(size_t size);
    
    /**
     * @brief 启用页式缓存管理
     * @param enable 是否启用
     * @param page_size 页大小（字节）
     */
    void enable_paged_attention(bool enable, size_t page_size = 4 * 1024);
    
    /**
     * @brief 获取页大小
     * @return 页大小（字节）
     */
    size_t get_page_size() const;
    
    /**
     * @brief 获取当前内存类型
     * @return 内存类型
     */
    MemoryType get_memory_type() const;
    
    // /**
    //  * @brief 切换内存类型
    //  * @param memory_type 新的内存类型
    //  * @return 是否切换成功
    //  */
    // bool switch_memory_type(MemoryType memory_type);
    
    /**
     * @brief 检查是否启用了页式缓存管理
     * @return 是否启用页式缓存管理
     */
    bool is_paged_attention_enabled() const;
    
    /**
     * @brief 获取指定请求的KV缓存数量
     * @param request_id 请求ID
     * @return KV缓存数量
     */
    size_t get_cache_count(const std::string& request_id) const;
    
    /**
     * @brief 获取所有请求ID列表
     * @return 请求ID列表
     */
    std::vector<std::string> get_all_request_ids() const;
};
