#include <vector>
#include <map>
#include <unordered_map>
#include <mutex>
#include <stdexcept>
#include <iostream>

// 内存(显存)池管理器
class PoolMemoryManager {
public:
    // 构造函数：获取上游管理器预分配的一大块内存
    explicit PoolMemoryManager(void* pool_ptr, size_t pool_size, size_t alignment = 256);

    // 析构函数：并不会释放整个内存池，释放操作由其上游管理器释放
    ~PoolMemoryManager();
    
    // // 禁用拷贝和赋值
    // PoolMemoryManager(const PoolMemoryManager&) = delete;
    // PoolMemoryManager& operator=(const PoolMemoryManager&) = delete;

    // 核心分配逻辑
    void* allocate(size_t size);
    
    // 核心回收逻辑
    void deallocate(void* ptr);

    size_t get_free_memory() const;

    size_t get_total_memory() const;

private:
    // 辅助函数
    size_t get_aligned_size(size_t size) const;
    void add_free_block(void* ptr, size_t size);
    void remove_free_block(void* ptr, size_t size);

    // 数据成员
    void* pool_ptr_ = nullptr;        // 指向内存池起始位置
    size_t pool_size_;                // 内存池总大小
    size_t alignment_;                // 内存对齐字节数

    // 管理空闲块 (按大小排序，用于 allocate)
    std::multimap<size_t, void*> free_blocks_by_size_;
    
    // 管理空闲块 (按地址排序，用于 deallocate 合并)
    std::map<void*, size_t> free_blocks_by_addr_;
    
    // 管理已分配块 (地址 -> 大小)
    std::unordered_map<void*, size_t> allocated_blocks_;

    size_t total_free_memory_ = 0;
    mutable std::mutex mutex_;        // 保证线程安全
};
